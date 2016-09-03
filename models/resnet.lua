--[[
This code is part of Ultrasound-Nerve-Segmentation using Torchnet

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Loading U-Net model
--]]

require 'nn'
require 'nngraph'
require 'cunn'

local MSRInit = require 'models/initialization.lua'

local MaxPooling = nn.SpatialMaxPooling
local Convolution = nn.SpatialConvolution
local BatchNorm = nn.SpatialBatchNormalization
local UpConvolution = nn.SpatialFullConvolution
local Identity = nn.Identity
local Add = nn.CAddTable
local ReLU = nn.ReLU
local Avg = cudnn.SpatialAveragePooling

-- The shortcut layer is either identity or 1x1 convolution
local function shortcut(nInputPlane, nOutputPlane, stride)
	local useConv = shortcutType == 'C' or
	 	(shortcutType == 'B' and nInputPlane ~= nOutputPlane)
	if useConv then
	 -- 1x1 convolution
		return nn.Sequential()
			:add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
			:add(SBatchNorm(nOutputPlane))
	elseif nInputPlane ~= nOutputPlane then
	 -- Strided, zero-padded identity shortcut
	 	return nn.Sequential()
			:add(nn.SpatialAveragePooling(1, 1, stride, stride))
			:add(nn.Concat(2)
		   		:add(nn.Identity())
		   		:add(nn.MulConstant(0)))
	else
	 	return nn.Identity()
	end
end



local function basicblock(nIn, nOut, stride)
	local kW, kH, padW, padH = 3, 3, 1, 1 -- parameters for 'same' conv layers

	local net = nn.Sequential()
	
	net:add(Convolution(nIn, nOut, kW, kH, stride, stride, padW, padH))
	net:add(BatchNorm(nOut))
	net:add(ReLU(true))
	net:add(Convolution(nOut, nOut, kW, kH, 1, 1, padW, padH))
	net:add(BatchNorm(nOut))

	return nn.Sequential()
		:add(nn.ConcatTable()
			:add(net)
			:add(shortcut(nIn, nOut, stride)))
		:add(Add(true))
		:add(ReLU(true))
end

local function layer(block, inFeatures, outFeatures, count, stride)
	local s = nn.Sequential()
	local n_in = inFeatures
	for i=1,count do
		if i > 1 then 
			n_in = outFeatures 
		end
		s:add(block(n_in, outFeatures, i == 1 and stride or 1))
	end
	return s
end

--- Returns model, name which is used for the naming of models generated while training
function createModel(opt)
	opt = opt or {}
	local nbClasses = opt.nbClasses or 2 	-- # of labls
	local nbChannels = opt.nbChannels or 1 	-- # of labls

	local model = nn.Sequential()
	model:add(Convolution(nbChannels, 32, 3, 3, 1, 1, 1, 1))
	model:add(BatchNorm(32))
	model:add(ReLU(true))
	model:add(layer(basicblock,  32,  64, 2, 1))
	model:add(layer(basicblock,  64, 128, 2, 1))
	model:add(layer(basicblock, 128, 256, 2, 1))
	model:add(layer(basicblock, 256, 512, 2, 1))
	model:add(Avg(3, 3, 1, 1, 1, 1))
	model:add(Convolution(512, nbClasses, 1,1))

	MSRInit(model)

	return model, 'resnet_512'

end
