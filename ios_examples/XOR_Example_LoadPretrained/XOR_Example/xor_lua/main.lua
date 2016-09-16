torch.setdefaulttensortype('torch.FloatTensor')

require('util')
model = require('johnson')

function loadNeuralNetwork(path)
    print (path)
    loadParams(model, path)
    print ("Loaded Neural Network -- Success")
    model:evaluate()
    input = torch.FloatTensor(1, 3, 256, 256)
    model:forward(input)
    opts = {inplace=true, mode='inference'}
    local mems1 = optnet.countUsedMemory(model)
    optnet.optimizeMemory(model, input, opts)
    local mems2 = optnet.countUsedMemory(model)
    print("Original net size:", mems1.total_size/1024/1024)
    print("Optimized net size:", mems2.total_size/1024/1024)
    print ("MemOptimized Neural Network -- Success")
    print ("Model Architecture --\n")
    print (model)
    print ("---------------------\n")
end

function classifyExample(tensorInput, tensorOutput, path)
    tensorOutput:copy(model:forward(tensorInput))
    return 1
end