import * as tf from '@tensorflow/tfjs-node'
import model from "./classes/model"

//テスト
//const $m=model.load("test").then(a=>console.dir(a));
//console.dir($m);
const $m = new model({
    path: "test01",
    labels: ["CAT", "DOG"],
    imageShape: [97, 97, 1]
});
function random($n:number):number{

    return Math.ceil(Math.random()*$n);
}



async function TrainLoop(last:number=100){
  
const $imageBuff: tf.Tensor3D[] = [];
const $labelBuff: tf.Tensor[] = [];

//トレーニングデータを準備
for (let i = 1; i <= 5000; i++) {
    var $image = model.image2tensor(`/mnt/ssd1/10_画像素材/dogs_vs_cats/PetImages/Cat/${i}.jpg`);
    $imageBuff.push($image);
    $labelBuff.push(model.buildLabel(0,2));
}
for (let i = 1; i <= 5000; i++) {
    var $image = model.image2tensor(`/mnt/ssd1/10_画像素材/dogs_vs_cats/PetImages/Dog/${i}.jpg`);
    $imageBuff.push($image);
    $labelBuff.push(model.buildLabel(1,2));
}

const $images = tf.stack($imageBuff);
const $labels = tf.stack($labelBuff);



  if(last <= 0 )return;
  await $m.Train($images, $labels)
    .then(() => {
        console.log("Trained");
        var $test_image = model.image2tensor(`/mnt/ssd1/10_画像素材/dogs_vs_cats/PetImages/Dog/3000.jpg`);
        
        const answer = $m.Predict($test_image);
        console.log(`${answer=='DOG'?'o':'x'} 正:DOG 答:${answer}`);
        if (answer == 'DOG') {
        }
        else {

        }
        $m.save();
        return TrainLoop(last-1);
     });
}
TrainLoop(1);
//console.log( ( model.buildLabel(0,2) as tf.Tensor ).print() );
