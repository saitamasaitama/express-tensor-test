import express from 'express';
import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node'
import Image2Vec from './layers/Image2Vec';

function dd($obj:any){
  console.dir($obj);
  console.log(typeof($obj));
  console.log($obj.constructor.name);
  process.exit();
}

const L1=tf.sequential({
    layers:[
      tf.layers.dense({units:1, inputShape:[2]})
    ]
  });


const P= L1.predict(tf.tensor([
  [1,2],
  [3,4],
  [5,6],
  [7,8],
]));

(P as tf.Tensor).print();
dd(P);

//dd(
//  l1
//);

