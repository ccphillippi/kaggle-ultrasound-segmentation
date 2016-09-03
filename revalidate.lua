require 'image';
require 'paths';
require 'dataloader'
require 'machine'
require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'utils/utils'
require 'cutorch'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 context encoder training script')
cmd:text()
cmd:text('Options:')
cmd:option('-dataset','data/train.h5','Training dataset to be used')
--cmd:option('-model','models/unet','Path of the model to be used')
cmd:option('-modelDir', 'data/better_models', 'Saved model directory')
--cmd:option('-trainSize',100,'Size of the training dataset to be used, -1 if complete dataset has to be used')
cmd:option('-valSize',-1,'Size of the validation dataset to be used, -1 if complete validation dataset has to be used')
--cmd:option('-trainBatchSize',64,'Size of the batch to be used for training')
cmd:option('-valBatchSize',1,'Size of the batch to be used for validation')
--cmd:option('-savePath','data/saved_models/','Path to save models')
--cmd:option('-optimMethod','sgd','Algorithm to be used for learning - sgd | adam')
--cmd:option('-learningRate', 0.1,'Initial learning rate to be used with sgd or adam')
--cmd:option('-maxepoch',250,'Epochs for training')
cmd:option('-cvParam',2,'Cross validation parameter used to segregate data based on patient number')
cmd:option('-validOnly', true, 'Use the validation set only')
cmd:option('-device', 1, 'Which GPU to use')
cmd:option('-segmentationProb',.98, 'Probability required to label segment')

function Evaluator()
    return function(network, sample)
        network:forward(sample.input:cuda())
        return sample.input, sample.target, GetMaskProbabilities(network.output)
    end
end

function loadModel(path)
	return torch.load(path):cuda()
end

function revalidate(opt)

	cutorch.setDevice(opt.device)

	local dl = DataLoader(opt)
	local N = opt.valSize
	local valid_set = dl:GetData('val', N)
	local it = getIterator('test', valid_set, opt.valBatchSize)
	baseSegmentationProb = opt.segmentationProb
	local evaluate = Evaluator()
	local modelDir = paths.concat(opt.modelDir, tostring(opt.cvParam))

	local n_mask = 0
	local n_not_mask = 0
	local n_final = 0
	local dice_final = 0
	for f in paths.iterfiles(modelDir) do
		f = paths.concat(modelDir, f)
		net = loadModel(f)
		net:evaluate()

		local n = 0
		local total_dice = 0
		for sample in it() do
			local input, target, softmax = evaluate(net, sample)
			local dice = CalculateDiceScore(net.output, target)

			n_mask = n_mask + target:gt(0):sum()
			n_not_mask = n_not_mask + target:lt(1):sum()

			total_dice = total_dice + dice
			dice_final = dice_final + dice
			n = n + 1
			n_final = n_final + 1
		end

		print(("%s: %.4f"):format(f, total_dice / n))
	end

	print(("Overall: %.4f"):format(dice_final / n_final))
	print(('Mask Ratio: %.4f'):format(n_mask / (n_mask + n_not_mask)))
end

local opt = cmd:parse(arg or {})
revalidate(opt)