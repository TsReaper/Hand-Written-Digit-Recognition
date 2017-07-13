'use strict';

class CanvasWrapper
{
  constructor(canvas, enabled = false)
  {
    // Bind this
    this.getGrayImage = this.getGrayImage.bind(this);
    this.setGrayImage = this.setGrayImage.bind(this);
    this.clear = this.clear.bind(this);
    this.enable = this.enable.bind(this);
    this.mouseDown = this.mouseDown.bind(this);
    this.mouseUp = this.mouseUp.bind(this);
    this.mouseMove = this.mouseMove.bind(this);
    
    this.canvas = canvas;
    if (enabled)
      this.enable();
  }
  
  getGrayImage()
  {
    let canvas = this.canvas;
    const image = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
    let ret = [];
    
    for (let i = 0; i < image.data.length; i += 4)
      ret.push(image.data[i + 3]);
    
    return ret;
  }
  
  setGrayImage(gray, w, h)
  {
    let canvas = this.canvas;
    let image = canvas.getContext('2d').createImageData(canvas.width, canvas.height);
    
    // Scale grayscale image to fit canvas
    let pos = 0;
    for (let i = 0; i < canvas.height; i++)
      for (let j = 0; j < canvas.width; j++)
      {
        const ii = Math.floor(i / canvas.height * h);
        const jj = Math.floor(j / canvas.width * w);
        const pp = ii * w + jj;
        image.data[pos] = image.data[pos + 1] = image.data[pos + 2] = 0;
        image.data[pos + 3] = gray[pp];
        pos += 4;
      }
    
    canvas.getContext('2d').putImageData(image, 0, 0);
  }
  
  clear()
  {
    let canvas = this.canvas;
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  }
  
  enable()
  {
    this.canvas.addEventListener('mousedown', this.mouseDown);
    this.canvas.addEventListener('mouseup', this.mouseUp);
    this.canvas.addEventListener('mouseout', this.mouseUp);
    this.canvas.addEventListener('mousemove', this.mouseMove);
  }
  
  disable()
  {
    this.canvas.removeEventListener('mousedown', this.mouseDown);
    this.canvas.removeEventListener('mouseup', this.mouseUp);
    this.canvas.removeEventListener('mouseout', this.mouseUp);
    this.canvas.removeEventListener('mousemove', this.mouseMove);
  }
  
  mouseDown(e)
  {
    const rect = this.canvas.getBoundingClientRect();
    this.lastMouseX = e.clientX - rect.left;
    this.lastMouseY = e.clientY - rect.top;
    
    // Only triggers when left clicked
    if (e.button == 0)
      this.isMouseDown = true;
  }
  
  mouseUp(e)
  {
    this.isMouseDown = false;
  }
  
  mouseMove(e)
  {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    if (this.isMouseDown)
    {
      let ctx = this.canvas.getContext('2d');
      ctx.beginPath();
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 15;
      
      ctx.moveTo(this.lastMouseX, this.lastMouseY);
      ctx.lineTo(mouseX, mouseY);
      ctx.lineJoin = ctx.lineCap = 'round';
      ctx.stroke();
    }
    
    this.lastMouseX = mouseX;
    this.lastMouseY = mouseY;
  }
}
