torch.setdefaulttensortype('torch.FloatTensor')

require('util')
model = require('johnson')

function loadNeuralNetwork(path)
    print (path)
    loadParams(model, path)
    model:evaluate()
    print ("Loaded Neural Network -- Success")
    print ("Model Architecture --\n")
    print (model)
    print ("---------------------\n")
end

function classifyExample(tensorInput, tensorOutput, path)
    torch.save(path .. "input.t7", tensorInput)
    tensorOutput:copy(model:forward(tensorInput))
    print(path)
    torch.save(path .. "output.t7", tensorOutput)
    return 1
end