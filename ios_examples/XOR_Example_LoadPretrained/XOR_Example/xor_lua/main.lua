torch.setdefaulttensortype('torch.FloatTensor')

params_mode = 'style'
params_tv_weight = 0
model = require('johnson')

--model = ""

function loadNeuralNetwork(path)
    print (path)
    print ("Loaded Neural Network -- Success")
    --model = torch.load(path)

    print ("Model Architecture --\n")
    print (model)
    print ("---------------------\n")
end

function classifyExample(tensorInput)
    --v = model(tensorInput)
    input = torch.FloatTensor(1, 3, 256, 256)
    output = model:forward(tensorInput)
    --print(output)
    return 1
end