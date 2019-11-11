import express from 'express';
import multer from 'multer';
import * as fs from 'fs'
import * as tf from '@tensorflow/tfjs-node'
import { selu, Tensor } from '@tensorflow/tfjs-node';
import routes from './routes'
import simulator from './classes/modelSimulator'

require('@tensorflow/tfjs-node-gpu');
const port = 3000;

/**
 * デバッグ用
 */
function dd($obj:any){
  console.dir($obj);

}

/**
 * 画像関連処理用
 */

function image2tensor(
  $file: string,
  $resize:[number,number]=[97,97]
): tf.Tensor3D {
  try{
  const $image= tf.node.decodeImage(
      fs.readFileSync($file)
  ) as tf.Tensor3D;
  return $image;
 }catch($e){
    console.log(`E:${$file}`);
    return tf.zeros([97,97,1]);
  }
}

interface fromto{
  from:Tensor,
  to:Tensor,
}

const app = express();
const upload=  multer({ dest: './uploads/' });
app.set('view engine', 'pug');
app.use(express.static('public'));

/*
ルーター読み込み
*/
Object.keys(routes).forEach(k=>{
  console.log(`BEFORE USE ${k}`);
  console.dir(routes[k]);
  app.use(`/${k}`,routes[k]);
});


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


app.get('/test',(req,res)=>res.render("include/2dA",{
  tensor:tf.ones([3,4,5])
}));

app.get('/sim',(req,res)=>{

  const sim:simulator = new simulator(tf.sequential());
  return res.render("simulate",{
    sim:sim
  });
});


app.post('/file-up', upload.single('image'), (req, res, next) => {

  const tensor=tf.image.resizeBilinear(image2tensor(req.file.path),[60,60]);

  const $input=tf.input({shape:tensor.shape});
  const gray= tf.layers.dense({
    inputShape:tensor.shape,
    units:1,
    useBias:true,
    weights:[
      tf.ones([3,1]).div(3),
      tf.ones([1]).div(3),
    ],
    activation:'relu',
    batchSize:255,
  });

  const gray2=tf.layers.conv2d({
    inputShape:tensor.shape,
    kernelSize:2,
    filters:1,
    weights:[
      tf.ones([2,2,3,1]).div(12)
    ],
    batchSize:255,
    useBias:false,
    activation:'relu',
    padding:'same'
  });
  const avg=tf.layers.averagePooling2d({
    inputShape:tensor.shape,
    poolSize:[3,3],
    strides:1,
  });

  const max=tf.layers.maxPooling2d({
    inputShape:tensor.shape,
    poolSize:[2,2],
    strides:1,
  });

  console.dir(gray2.apply(tf.stack([tensor])));

  return res.render("include/2dA",{
    req:req.file,
    images:[
      tensor,
      gray.apply(tensor),
      gray2.apply(tf.stack([tensor])),
      avg.apply(tf.stack([tensor])),
      
      max.apply(tf.stack([tensor])),
    ],
    tensor:tensor
  });
});
