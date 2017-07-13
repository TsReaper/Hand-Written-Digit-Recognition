'use strict';

class Network
{
  constructor(json)
  {
    // Bind this
    this.predict = this.predict.bind(this);
    
    this.size = json.size;
    this.weights = json.weights;
    this.biases = json.biases;
  }
  
  predict(input)
  {
    let a = input.slice();
    let z;
    
    for (let i = 0; i < this.size.length-1; i++)
    {
      z = new Array(this.size[i+1]).fill(0);
      for (let j = 0; j < z.length; j++)
      {
        for (let k = 0; k < a.length; k++)
          z[j] += this.weights[i][j][k] * a[k];
        z[j] += this.biases[i][j];
        z[j] = sigmoid(z[j]);
      }
      a = z.slice();
    }
    
    return a;
  }
}

function sigmoid(x)
{
  return 1.0 / (1.0 + Math.exp(-x));
}
