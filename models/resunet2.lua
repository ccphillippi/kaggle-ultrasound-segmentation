--[[
This code is part of Ultrasound-Nerve-Segmentation using Torchnet

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Loading U-Net model
--]]

require 'nn'
require 'nngraph'

local MSRInit = require 'models/initialization.lua'

local MaxPooling = nn.SpatialMaxPooling
local Convolution = nn.SpatialConvolution
local BatchNorm = nn.SpatialBatchNormalization
local UpConvolution = nn.SpatialFullConvolution
local Identity = nn.Identity
local Add = nn.CAddTable
local ReLU = nn.ReLU
local Dropout = nn.Dropout

--- Creates a conv layer given number of input feature maps and number of output feature maps, with dropout
-- @param nIn Number of input feature maps
-- @param nOut Number of output feature maps
local function ConvLayers(nIn, nOut)
	local kW, kH, dW, dH, padW, padH = 3, 3, 1, 1, 1, 1 -- parameters for 'same' conv layers

	local net = nn.Sequential()
	net:add(BatchNorm(nIn))
	net:add(ReLU(true))
	net:add(Convolution(nIn, nOut, kW, kH, dW, dH, padW, padH))

	net:add(BatchNorm(nOut))
	net:add(ReLU(true))
	net:add(Convolution(nOut, nOut, kW, kH, dW, dH, padW, padH))

	return net
end

--- Returns model, name which is used for the naming of models generated while training
function createModel(opt)
	opt = opt or {}
	local nbClasses = opt.nbClasses or 2 	-- # of labls
	local nbChannels = opt.nbChannels or 1 	-- # of labls

	local input = nn.Identity()()

	--local conv = Convolution(nbChannels, 32, 3, 3, 1, 1, 1, 1)(input)

	local D1 = ConvLayers(nbChannels,32)(input)
	local D2 = ConvLayers(32,64)(MaxPooling(2,2)(D1))
	local D3 = ConvLayers(64,128)(MaxPooling(2,2)(D2))
	local D4 = ConvLayers(128,256)(MaxPooling(2,2)(D3))

	local B = ConvLayers(256,512)(MaxPooling(2,2)(D4))

	local U4 = ConvLayers(256,256)(Add(true)({ D4, ReLU(true)(UpConvolution(512,256, 2,2,2,2)(B)) }))
	local U3 = ConvLayers(128,128)(Add(true)({ D3, ReLU(true)(UpConvolution(256,128, 2,2,2,2)(U4)) }))
	local U2 = ConvLayers(64,64)(Add(true)({ D2, ReLU(true)(UpConvolution(128,64, 2,2,2,2)(U3)) }))
	local U1 = ConvLayers(32,32)(Add(true)({ D1, ReLU(true)(UpConvolution(64,32, 2,2,2,2)(U2))   }))

	local net = nn.Sequential()
	net:add(nn.gModule({input}, {U1}))
	net:add(Convolution(32, nbClasses, 1,1))

	MSRInit(net)

	return net,'resunet2_512'
end
