import express from 'express';
import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node'

import Antirectifier from './custom_layer'

require('@tensorflow/tfjs-node-gpu');

function dd($obj:any){
  console.dir($obj);
  process.exit();
}

function file2Tensor($file:string):tf.Tensor3D{
  return tf.node.decodeImage(
    fs.readFileSync($file)
  ) as tf.Tensor3D;
}
const $image=file2Tensor("/mnt/nvme1/imas-cg/fffdb0dbb5c29d377363b4559084e8f3.jpg");
const $shape = $image.shape;

const $images= tf.stack([
  $image,
]);

const $input:tf.SymbolicTensor = tf.input({shape:
  [ 800, 640, 3 ]
});

const layers=tf.sequential();
layers.add(tf.layers.conv2d({
  inputShape:[ 800, 640, 3 ],
  filters:8,
  kernelSize:3,
  activation:'relu'
}));
layers.add(tf.layers.dense({
  units:2,
  activation:'relu'
}));
layers.add(tf.layers.dropout({
  rate:0.5
}));
layers.add(tf.layers.maxPooling2d({
  poolSize: [100,100] })
);
layers.add(tf.layers.maxPooling2d({
  poolSize: [3,2] })
);
//layers.add(new Antirectifier());
layers.add(tf.layers.flatten());

const model=tf.model({
  inputs:$input,
  outputs:layers.apply($input) as tf.SymbolicTensor
});


//console.log(model);
(model.predict(
  $images
)as tf.Tensor).print();

model.summary();

model.save('file://result.nn');
