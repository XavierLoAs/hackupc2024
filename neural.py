import numpy, random, os
lr = 1 #learning rate
bias = 1 #value of bias
weights = [random.random(),random.random(),random.random()] #weights generated in a list (3 weights in total for 2 neurons and the bias)
weightsSigmoid = [random.random(),random.random(),random.random()] #weights generated in a list (3 weights in total for 2 neurons and the bias)


def Perceptron(input1, input2, output) :
   outputP = input1*weights[0]+input2*weights[1]+bias*weights[2]
   if outputP > 0 : #activation function (here Heaviside) mirar Sigmoid bebé
      outputP = 1
   else :
      outputP = 0
   error = output - outputP #Calculamos el error del output esperado (lo ponemos nosotros) y el calculado a partir de los pesos
   weights[0] += error * input1 * lr
   weights[1] += error * input2 * lr #Actualizamos los pesos (multiplicamos por el lr, para ver cuanto modificamos en base al error)
   weights[2] += error * bias * lr
   
if __name__ == "__main__":
    for i in range(50000) :
        Perceptron(1,1,1) #True or true
        Perceptron(1,0,1) #True or false
        Perceptron(0,1,1) #False or true
        Perceptron(0,0,0) #False or false
    for weight in weights:
        print(weight)
    
    for weightS in weightsSigmoid:
        print(weightS)   
    x = int(input())
    y = int(input())
    outputP = x*weights[0] + y*weights[1] + bias*weights[2]
    if outputP > 0 : #activation function
        outputP = 1
    else :
        outputP = 0
    outputS = 1/(1+numpy.exp(-outputP)) #sigmoid function
    errorS = outputP - outputS
    print(x, "or", y, "is (outputP): ", outputP, " is (outputS)", outputS, "con error: ", errorS)