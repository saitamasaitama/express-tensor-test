import * as tf from '@tensorflow/tfjs-node'

interface GANOpts{
  name:string,
  label_size:number,
  output_shape:number[]
}

interface GANOut{
  generator :any,
  judger : any
}

function GAN(opts:GANOpts):GANOut
{
  const out_size=opts.output_shape.reduce((carry,item)=>carry*item,1);
  console.dir(out_size);
  
  const $g =tf.sequential({
    layers:[
      tf.layers.dense({
        units:out_size,
        inputShape: [ opts.label_size ],
        activation:'relu'
      })
      //最終レイヤ(x y zにする )

    ]
  });
  //画像を入れて判定する奴
  const $j =tf.sequential({
    layers:[
      tf.layers.dense({
        units:out_size,
        inputShape:opts.output_shape,
        activation:'relu'
      }),
      //最終レイヤ（denseに納める）
      tf.layers.dense({
        units:opts.label_size,
        activation:'softmax'
      })
    ]

  });

  //レイヤを定義


  //ラベルサイズから順に

  //$j.add(   ); 

  //出力サイズから順に構築



  return {
    generator:$g,
    judger:$j
  }
}

/*
export {
  model
}
*/

GAN({
   name:"a",
  label_size:3,
  output_shape:[20,20,3]
})
