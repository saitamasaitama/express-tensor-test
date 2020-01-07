import * as tf from '@tensorflow/tfjs-node'
import * as fs from 'fs';

function dd($obj: any) {
  console.dir($obj);
  process.exit();
}


/**
  特定サイズのラベルを作成する
  例： 1 , 5
  [0,1,0,0,5]
 */



interface modelOpts{
  path:string,
  labels:string[],
  imageShape:number[],
  layers?:tf.LayersModel,
}
//デフォルト値
const modelOptsDefault:modelOpts={
  path:"test",
  imageShape:[97,97,1],
  labels:["CAT","DOG"],
};

class model {

  public $savePath:string;
  public $inputShape:number[];
  public $model:tf.LayersModel;
  public $labels:string[];

  public constructor(opts:modelOpts=modelOptsDefault){

    if(!opts.layers){
      opts.layers=tf.sequential({
        layers:[
          tf.layers.inputLayer({
            inputShape:opts.imageShape
          }),
          //単色にする
   //       tf.layers.dense({ units: 1, activation: 'relu6' }),
          tf.layers.conv2d({
            filters:8,
            kernelSize:3,
            activation:'relu'
          }),
          tf.layers.conv2d({
            filters: 4,
            kernelSize: 3,
            activation: 'relu',
          }),
          tf.layers.maxPooling2d({ poolSize: [8, 8] }),
          tf.layers.flatten(),
          tf.layers.dropout({ rate: 0.25 }),
          tf.layers.dense({ units: 512, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.25 }),
          tf.layers.dense({
            units:opts.labels.length,
            activation:'softmax'
          })
        ]
      });
    }
    this.$savePath=opts.path;
    this.$labels = opts.labels;
    this.$inputShape=opts.imageShape;
    const $input =tf.input({shape:this.$inputShape});

    this.$model=tf.model({
      inputs:$input,
      outputs:opts.layers.apply($input)as tf.SymbolicTensor
    });


    this.$model.compile({
      optimizer: 'rmsprop',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  }

  public ToString(){ 
    return `MODEL` ;
  }

  public static Build(){
  }

  public static async load(path:string){

    return await tf.loadLayersModel(`file://${path}.nn/model.json`).then(
      layers =>{
        console.log("LOAD MODELS");
        return new model({
        path:path,
        labels:["CAT","DOG"],
        imageShape:[97,97,1],
        layers:layers
        });
      }
    );
  }

  public async Train(
    $images:tf.Tensor,
    $labels:tf.Tensor
    ){
    
    return await this.$model.fit(
      $images,
      $labels,
      {
        batchSize:9960,
        epochs:10,
        verbose:1
      }
    );
  }

  public Predict($image:tf.Tensor3D):string{
    const $tensor=(this.$model.predict(tf.stack([$image])) as tf.Tensor);
    console.log("PREDICT-RESULT!");
    console.log($tensor.print());
    const $result=$tensor.argMax(1).dataSync()[0];
    return this.$labels[$result];
  }

  public async save() {
    this.$model.save(`file://./${this.$savePath}.nn`);
  }

  public static image2tensor(
    $file: string,
    $resize:[number,number]=[97,97]
  ): tf.Tensor3D {
    try{
    const $image= tf.node.decodeJpeg(
        fs.readFileSync($file)
        ,1 //チャンネル数
        ,1 //縮小率
        ,true
        ,true
        ,1
    ) as tf.Tensor3D;
     const $resized= tf.image.resizeBilinear(
      $image,$resize
    )as tf.Tensor3D;
    return $resized.div(255);
   }catch($e){
      console.log(`E:${$file}`);
      return tf.zeros([97,97,1]);
    }

  }

  public static async tensor2jpeg(
      $image:tf.Tensor3D,
      $path:string = "test2.jpg"
      )
  {
    console.log("BEGIN File Write");

    await tf.node.encodeJpeg($image).then(
        b=>{
         fs.writeFileSync(`./public/${$path}`,b);
         console.log(`JPG ${$path} write OK!`);
        }
      );
  }


  public static buildLabel($num:number,$size:number):tf.Tensor{
    const arr=Array($size).fill(0);
    arr[$num]=1;
    return tf.tensor(arr);
  }

}

export default model;


