import * as tf from '@tensorflow/tfjs-node'

const $in=tf.input({shape:[1]});

const model= tf.model({
  inputs:$in,

  outputs: tf.sequential({
      layers:[
        tf.layers.inputLayer({inputShape:[1]}),
        tf.layers.dense({
          units:2,
          activation:'softmax'
        }),
      ]  
    }).apply(
      $in
  )as tf.SymbolicTensor
});

model.compile({
  optimizer: 'rmsprop',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

model.summary();

model.fit(tf.tensor([
 [1],
 [2],
 [3],
 [4],
 [5],
]),tf.tensor([
  [0.1,0.9],
  [0.3,0.7],
  [0.5,0.5],
  [0.7,0.3],
  [0.9,0.1],
]),{
  batchSize:100,
  epochs:1000,
  verbose:0
}).then(v=>{
  console.dir((model.predict(tf.tensor([2])) as tf.Tensor).dataSync() )
});

