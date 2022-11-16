import numpy as np
def gradientFunc(x):
    return 2 * x+6

def gradientDescent(gradient,start,learnrate,iter,tol=0.01):
    steps = [start]
    x = start
    
    for _ in range(iter):
        diff = -learnrate * gradient(x)
        if(np.abs(diff)<=tol):
            break

        x = x + diff
        
        steps.append(x)
    return(steps, x, learnrate)


history , result, learning_rate = gradientDescent(gradientFunc, 2, 0.1,100)
print("Steps: ", history)

print("Minimum ", result)

print("Learning Rate: ",learning_rate)

print("Iterations: ", len(history))