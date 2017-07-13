'use strict';

let input, processed;
let network = new Network(json);

function predict()
{
  const image = preprocess(input.getGrayImage(), 280);
  processed.setGrayImage(image, 28, 28);
  const res = network.predict(image.map(x => x / 255));
  
  let best = 0;
  for (let i = 1; i < 10; i++)
    if (res[i] > res[best])
      best = i;
  
  // Display result
  document.querySelector('#result').innerHTML = best.toString();
  document.querySelector('#res-confidence').innerHTML = `${Math.floor(res[best]*10000) / 100}% confidence`;
  
  for (let i = 0; i < 10; i++)
    document.querySelector('#res-' + i.toString()).innerHTML = i.toString() + ': ' + `${Math.floor(res[i]*10000) / 100}%`;
}

function clear()
{
  input.clear();
  processed.clear();
}

window.addEventListener('DOMContentLoaded', function(){
  let inputCanvas = document.querySelector('#input');
  input = new CanvasWrapper(inputCanvas, true);
  processed = new CanvasWrapper(document.querySelector('#processed'));
  predict();
  
  // Disable canvas right click menu
  inputCanvas.oncontextmenu = function(){
    return false;
  };
  
  inputCanvas.addEventListener('mousedown', function(e){
    if (e.button == 2)
      clear();
  });
  inputCanvas.addEventListener('mouseup', predict);
  inputCanvas.addEventListener('mouseout', predict);
  document.querySelector('#button').addEventListener('click', clear);
});