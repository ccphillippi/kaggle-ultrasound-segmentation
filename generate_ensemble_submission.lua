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
require 'utils/utils.lua'
require 'constants.lua'
require 'ensemble_submission'

torch.setnumthreads(1) -- Increase speed
torch.setdefaulttensortype('torch.FloatTensor')

-- command line instructions reading
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Submission file generation script')
cmd:text()
cmd:text('Options:')
cmd:option('-dataset','data/test.h5','Testing dataset to be used')
cmd:option('-ensembleDir','data/best_models','Path of the trained model directory to be used')
cmd:option('-csv','submisson.csv','Path of the csv file to be generated')
cmd:option('-testSize',5508,'Number of images for which data is to be generated - 5508 if all images on test set, 5635 if it is on train set')
cmd:option('-device', 1, 'GPU to run on')
cmd:option('-threshold',0.5,'Threshold of votes to accept pixel as nerve')
cmd:option('-combineSoftmax',true,'Combine softmaxes before taking threshold')


local opt = cmd:parse(arg or {}) -- Table containing all the above options
cutorch.setDevice(opt.device)
GenerateSubmission(opt)
