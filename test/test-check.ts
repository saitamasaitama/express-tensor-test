import * as tf from '@tensorflow/tfjs-node'
import model from "./classes/model"

//テスト
//const $m=model.load("test").then(a=>console.dir(a));
//console.dir($m);
model.load("test01")
.then(
    $m=>{
    console.log(TrainCheck($m,100));
    }
    );


function random($n:number):number{
  return Math.ceil(Math.random()*$n);
}

interface OKNG{
  OK:number,
  NG:number
}

function TrainCheck($m:model,last:number=100,carry:OKNG={
OK:0,
NG:0
}):OKNG{
  console.log("===>");
  if(last <= 0 )return carry;
  var r=random(10000);
  console.log(`>>>LOAD IMAGE ${r}  `);
  
  var correct= random(100)<50?"Cat":"Dog";

  var $test_image = model.image2tensor(`/mnt/ssd1/10_画像素材/dogs_vs_cats/PetImages/${correct}/${r}.jpg`);

  

  const answer = $m.Predict($test_image);
  console.log(`正:${correct.toUpperCase()} 答:${answer}`);
  console.log("<===");
  if (answer == correct.toUpperCase()) {
    carry.OK++;
  }
  else {
    carry.NG++;
  }

  return TrainCheck($m,last-1,carry);

}

