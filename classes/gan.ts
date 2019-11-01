import * as tf from '@tensorflow/tfjs-node'

const gen = tf.sequential({
  layers:[
    tf.layers.inputLayer({inputShape:[2,2,2]}),
    tf.layers.dense({units:4}),
  ]
});


const result=gen.predict(tf.ones([1,2,2,2]) ) as tf.Tensor;

result.print();
