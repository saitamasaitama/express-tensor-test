import * as models from "./model";
import * as image from "../../classes/image";
import * as tf from '@tensorflow/tfjs-node'

//モデル作成

/*const shape:number[]=[
  1,2,3 
];
*/

interface Tensorable{
  ToTensor () : tf.Tensor ;
}

const $model=tf.sequential({
layers:[
  tf.layers.inputLayer({
    inputShape:[4]
  }),
  tf.layers.dense({
    units:2,
    activation:'softmax'
  }),

]});
$model.compile({
  optimizer: 'rmsprop',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});
//モデル作成　ここまで

/*
画像クラス作成 ここから
*/
class data implements Tensorable {
  ToTensor(){
   return tf.tensor([1,2,3,4]);
 }
}

class label implements Tensorable {
  ToTensor(){
    return tf.tensor([1,0]);
  }
}

async function GanLearn
<
DATA extends Tensorable,
LABEL extends Tensorable
>(
      $model:tf.LayersModel,
      $datas:Array<DATA> =[],
      $labels:Array<LABEL>  =[]
        )
        {
  
  console.log("fit go!");
  //スタックにする
  const datas:any[]=[];
  $datas.forEach(v=>datas.push(v.ToTensor()));

  const labels:any[]=[];
  $labels.forEach(v=>labels.push(v.ToTensor()));

  return await $model.fit(
    tf.stack(datas),
    tf.stack(labels),
    {
      epochs:10,
      batchSize:2000
    }
  );
}

function GanModel(){

}

(async (path:string) => {
    console.dir(path);
    console.log("READY LEARN!");
    const out=await GanLearn(
      $model,
      [
        new data(),
        new data()
      ],
      [
        new label(),
        new label()
      ]
    );
    console.log("DONE LEARN!");
    console.dir(out);
    console.log("READY PREDICT...");
    ($model.predict(
    tf.tensor([
      [2,3,4,5]
    ])  
      ) as tf.Tensor
    ).print();
})(process.argv[2]);
