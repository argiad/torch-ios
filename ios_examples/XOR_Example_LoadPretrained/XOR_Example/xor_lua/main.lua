torch.setdefaulttensortype('torch.FloatTensor')

require('util')

params_mode = 'style'
params_tv_weight = 0
model = require('johnson')

function loadNeuralNetwork(path)
    print (path)
    loadParams(model, path)
    print ("Loaded Neural Network -- Success")
    print ("Model Architecture --\n")
    print (model)
    print ("---------------------\n")
end

function classifyExample(tensorInput, tensorOutput, path)
    tensorOutput:copy(model:forward(tensorInput):resizeAs(tensorInput))
    print(path)
    torch.save(path, tensorOutput)
    --print(tensorOutput)
    --print(tensorInput)
    return 1
end