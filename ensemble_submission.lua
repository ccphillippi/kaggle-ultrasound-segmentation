--[[
This code is part of Ultrasound-Nerve-Segmentation using Torchnet

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Generates submission file
--]]

require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'paths'
require 'image'
require 'hdf5'
require 'xlua'
require 'nngraph'
require 'csvigo'
require 'utils/utils'
require 'constants'

torch.setnumthreads(1) -- Increase speed
torch.setdefaulttensortype('torch.FloatTensor')

function do_nothing(input, target)
	return input, target
end

-- Returns table of transformations
function GetTransformations()
	-- flag to check if transformations to be reapplied on label, set true for segmentation
	local transformations = {}
	for i=1,2 do
		transformations[i] = {}
	end
	transformations[1]['do'],transformations[1]['undo'] = HorizontalFlip()
	--transformations[2]['do'],transformations[2]['undo'] = VerticalFlip()
	--transformations[3]['do'],transformations[3]['undo'] = Rotation(1)
	--transformations[4]['do'],transformations[4]['undo'] = Rotation(-1)
	transformations[2]['do'],transformations[2]['undo'] = do_nothing, do_nothing

	return transformations
end

--- Returns generated masks given the model, dataset, baseProbability and testSize
-- @param opt A table that contains path for the model, dataset and testSize
function GenerateMasks(opt)
	print("Loading dataset")
	local xf = hdf5.open(opt.dataset)
	local testImages = xf:read('/images'):all()
	xf:close()
	local masks = torch.zeros(opt.testSize,trueHeight*trueWidth)
	print("Generating masks")
	local maskCount = 0
	for i=1,opt.testSize do
		-- scale the image and divide the pixel by 255
		--local input = image.scale(testImages[i][1], imgWidth, imgHeight, interpolation)
		--print(("Generating mask for image(%d)"):format(i))
		local modelOutput = GetSegmentationModelOutputs(opt.ensembleDir,testImages[i][1],opt.threshold,opt.combineSoftmax)
		masks[i] = modelOutput:t():reshape(trueWidth*trueHeight) -- taking transpose and reshaping it for being able to convert to RLE
		if GetLabel(masks[i]) == 2 then
			maskCount = maskCount + 1
		end
		xlua.progress(i,opt.testSize)
	end
	print(("Number of images with masks : %d"):format(maskCount))
	return masks
end

--- Returns the mask after taking average over augmentation of images
-- @param model Model dir to be used
-- @param img Image to be used
function GetSegmentationModelOutputs(modelDir,img,threshold,softmax)
	local transformations = GetTransformations()
	local softTargetSums = torch.FloatTensor(trueHeight,trueWidth):fill(0)
	local predictionCounts = torch.FloatTensor(trueHeight,trueWidth):fill(0)


	heightGrid = {0, 18, 36}
	widthGrid = {0, 49, 98, 147, 196}
	size = imgHeight * 3

	--print('Running through models')
	local j = 1
	for folder in paths.iterdirs(modelDir) do
		folder = paths.concat(modelDir, folder)
		for f in paths.iterfiles(paths.concat(modelDir, folder)) do
			f = paths.concat(folder, f)

			--print(('\tEvaluating %s'):format(f))
			model = torch.load(f):cuda()
			model:evaluate()
			for _, x in ipairs(widthGrid) do
				for _, y in ipairs(heightGrid) do
					--print(("\t\tRunning transforms with offset(%d,%d)"):format(x, y))
					use_img = image.scale(
						image.crop(img, x+1, y+1, x+size, y+size),
						imgWidth, imgHeight,
						interpolation
					)
					for i=1,#transformations do
						local slice_output = GetMaskFromOutput(
							model:forward(
								transformations[i]['do'](use_img):reshape(1,1,imgHeight,imgWidth):cuda()
							)[1],
							true,
							transformations[i]['undo'],
							softmax
						)
						softTargetSums:sub(y+1, y+size, x+1, x+size):add(slice_output:float())
						predictionCounts:sub(y+1, y+size, x+1, x+size):add(1.)
						j = j + 1
					end
				end
			end
		end
	end
	softTargetSums:cdiv(predictionCounts)
	if not threshold then
		return softTargetSums
	end
	return GetTunedResult(softTargetSums,threshold)
end

--- Generates CSV given the masks and opt table containing csv path
function GenerateCSV(opt,masks)
	print("Generating RLE")
	-- rle encoding saved here, later written to csv
	local rle_encodings = {}
	rle_encodings[1] = {"img","pixels"}
	for i=1,opt.testSize do
		rle_encodings[i+1]={tostring(i),getRle(masks[i])}
		xlua.progress(i,opt.testSize)
	end
	-- saving the csv file
	csvigo.save{path=opt.csv,data=rle_encodings}
end

--- The main function that directs how movie is made
function GenerateSubmission(opt)
	cutorch.setDevice(opt.device)
	local masks = GenerateMasks(opt)
	GenerateCSV(opt,masks)
end
