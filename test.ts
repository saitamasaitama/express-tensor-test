import * as tf from '@tensorflow/tfjs-node'

const m =tf.sequential({
  layers:[
    tf.layers.inputLayer({inputShape:[1,2]})
  ]
});

async function load(path:string){
  return await tf.loadLayersModel("file://./test01.nn/model.json");
}

console.dir(tf.tensor([1,2,3]));
