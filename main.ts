import express from 'express';
import * as tf from '@tensorflow/tfjs-node'
require('@tensorflow/tfjs-node-gpu');
const port = 3000;

const app = express();
app.set('view engine', 'pug');
app.use(express.static('public'));


var $model:tf.LayersModel;

const $inputShape:number[] = [32,32,3];
const $input:tf.SymbolicTensor = tf.input({shape:$inputShape});
const $layers:tf.layers.Layer[]=[
      tf.layers.dense({
        units:12,
        name:"Default",
        activation:'relu'
      })
    ];
const OUT="";

var $message="特に無し";
function modelReBuild(){
  const result=tf.sequential({
    layers:[
      tf.layers.inputLayer({inputShape:$inputShape})
    ]
  });

  $layers.forEach($layer=>{
    console.log(`LAYER-Add:${$layer.name}`);
    result.add($layer);
  });

  return tf.model({
    inputs:$input,
    outputs:result.apply($input) as tf.SymbolicTensor
    });
}

app.get('/', (req:any, res:any)=>{
   return res.render('index',{
    inputs:$inputShape,
    layers:$layers,
    model:$model,
    message:$message,
   })
});

app.get('/dump', (req:any, res:any)=>{
   $model=modelReBuild();
   return res.send(
    ($model.predict(tf.ones([1,2])) as tf.Tensor).toString()
  );
});
app.post('/reset', (req:any, res:any)=>{
  $layers.length=0;
  $model=modelReBuild();
  return res.redirect('/');
});

app.post('/', (req:any, res:any)=>{
    $layers.push(tf.layers.dense({
      units:1,
      name:"AddLayer"+Math.random().toString(36).slice(-8)
    }));
    //便宜的にモデルを再構築
    console.log("モデル再構築:");
    const d=new Date();
     $message="モデル再構築"+d;
    $model=modelReBuild();
    
    //再学習
    return res.redirect('/');
  }
);



//実行
app.post('/run',(req:any,res:any)=>{});

app.post('/layer-add',(req:any,res:any)=>{
  $layers.push(tf.layers.dense({
    units:1,
    name:"AddLayer"
  }));
  return res.redirect("/");
});
app.listen(port,() => console.log(`ポート${port}番で処理開始`));


