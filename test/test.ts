import express from 'express';
import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node'
//import model from "./classes/model"

function dd($obj: any) {
  console.dir($obj);
  process.exit();
}



  const $in= tf.input({shape:[10]});


  const layers=tf.sequential({
    layers:[
        // 1-10の配列を入れて
        tf.layers.inputLayer({inputShape:src.shape}),
        //画像にする
//        tf.layers.repeatVector()     
        tf.layers.upSampling2d({
        })    
//        tf.layers.dense({units:3,activation:'sigmoid'})
    ]
    
  });

  const $input=tf.input({shape:src.shape});
  const model=tf.model({
    inputs:$input,
    outputs:layers.apply($input)as tf.SymbolicTensor
  });

  model.compile({optimizer:'sgd',loss:'meanSquaredError'});

  console.log("meron");
  (model.predict(tf.stack([src])) as tf.Tensor).print();



