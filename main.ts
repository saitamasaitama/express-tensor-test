import express from 'express';
import * as tf from '@tensorflow/tfjs-node'
require('@tensorflow/tfjs-node-gpu');
const port = 3000;
tf.fromPixels(info.data);

const app = express();
app.set('view engine', 'pug');
app.use(express.static('public'));

console.dir(tf.fromPixels('image'));

var model:tf.LayersModel|undefined=undefined;

const $inputShape:number[] = [2];
const $input:tf.SymbolicTensor = tf.input({shape:$inputShape});
const $layers:tf.layers.Layer[]=[];

const OUT="";

function modelReBuild(){
  const result=tf.sequential();
  result.add(tf.layers.inputLayer({
    inputShape:$inputShape
  }));
  result.add(tf.layers.dense({
    units:4
  }));


  return tf.model({
    inputs:$input,
    outputs:result.apply($input) as tf.SymbolicTensor
    });
}

app.get('/', (req:any, res:any)=>{
   return res.render('test',{})
});

app.get('/dump', (req:any, res:any)=>{
   model=modelReBuild();
   return res.send(model);
   return res.send(
    (model.predict(tf.ones([1,2])) as tf.Tensor).toString()
  );
});

app.post('/', (req:any, res:any)=>{
    //便宜的にモデルを再構築
    console.log("モデル再構築:");  
    //再学習
    return res.render('test',{});
  }
);


//パラメータチェンジ

//実行
app.post('/run',(req:any,res:any)=>{});
app.listen(port,() => console.log(`ポート${port}番で処理開始`));


//(model.predict(tf.ones([1,2])) as tf.Tensor).print();
//model.pred

