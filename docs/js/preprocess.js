'use strict';

function cutImage(image, size)
{
  // Find boundary
  let minI = size, minJ = size, maxI = 0, maxJ = 0;
  
  for (let i = 0; i < size; i++)
    for (let j = 0; j < size; j++)
      if (image[i*size + j] > 0)
      {
        minI = Math.min(minI, i);
        minJ = Math.min(minJ, j);
        maxI = Math.max(maxI, i);
        maxJ = Math.max(maxJ, j);
      }
  
  // Cut image
  let newSize = Math.max(maxI - minI + 1, maxJ - minJ + 1);
  let ret = new Array(newSize * newSize).fill(0);
  for (let i = 0; i < newSize; i++)
    for (let j = 0; j < newSize; j++)
    {
      if (i+minI >= size || j+minJ >= size)
        ret[i*newSize + j] = 0;
      else
        ret[i*newSize + j] = image[(i+minI)*size + (j+minJ)];
    }
  
  return [ret, newSize];
}

function shrinkImage(image, oldSize, newSize)
{
  let ret = new Array(newSize * newSize).fill(0);
  let cnt = new Array(newSize * newSize).fill(0);
  
  for (let i = 0; i < oldSize; i++)
    for (let j = 0; j < oldSize; j++)
    {
      const ii = Math.floor(i / oldSize * newSize);
      const jj = Math.floor(j / oldSize * newSize);
      const pp = ii * newSize + jj;
      ret[pp] += image[i*oldSize + j];
      cnt[pp]++;
    }
  
  for (let i = 0; i < ret.length; i++)
    ret[i] = cnt[i] > 0 ? Math.round(ret[i] / cnt[i]) : 0;
  
  return ret;
}

function transformImage(image, oldSize, newSize)
{
  // Calculate mass center
  let centerI = 0, centerJ = 0;
  let tot = 0;
  
  for (let i = 0; i < oldSize; i++)
    for (let j = 0; j < oldSize; j++)
    {
      const pixel = image[i*oldSize + j];
      centerI += i * pixel;
      centerJ += j * pixel;
      tot += pixel;
    }
  centerI = Math.round(centerI / tot);
  centerJ = Math.round(centerJ / tot);
  
  // Transform image
  const deltaI = Math.floor(newSize/2) - centerI;
  const deltaJ = Math.floor(newSize/2) - centerJ;
  let ret = new Array(newSize * newSize).fill(0);
  
  for (let i = 0; i < oldSize; i++)
    for (let j = 0; j < oldSize; j++)
      if (deltaI + i < newSize && deltaJ + j < newSize)
        ret[(deltaI+i)*newSize + (deltaJ+j)] = image[i*oldSize + j];
  
  return ret;
}

function preprocess(image, oldSize, newSize = 28, shrinkSize = 20)
{
  let t = cutImage(image, oldSize);
  return transformImage(shrinkImage(t[0], t[1], shrinkSize), shrinkSize, newSize);
}
