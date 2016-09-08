torch.setdefaulttensortype('torch.FloatTensor')

require('util')

params_mode = 'style'
params_tv_weight = 0
model = require('johnson')
inputImage = torch.FloatTensor(1, 3, 256, 256)

function loadNeuralNetwork(path)
    print (path)
    loadParams(model, path)
    print ("Loaded Neural Network -- Success")
    print ("Model Architecture --\n")
    print (model)
    print ("---------------------\n")
end

function loadImageFromDirectory(path)
    --inputImage = image.load(path, 3):float()
end

function classifyExample()
    input = torch.FloatTensor(1, 3, 256, 256)
    output = model:forward(input)
    return 1
end