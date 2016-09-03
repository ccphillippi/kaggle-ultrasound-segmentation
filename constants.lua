--[[
This code is part of Ultrasound-Nerve-Segmentation using Torchnet

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Constants used throughout the code
--]]

require "threshold"

imgHeight = 128 -- Height of the image to be resized to for use
imgWidth = 128 -- Width of the image to be resized to for use
trueHeight = 420 -- True height of the image
trueWidth = 580 -- True width of the image
nbClasses = 2 -- Number of classes in output, 2 since mask/no mask
interpolation = 'bicubic' -- Interpolation to be used for resizing
