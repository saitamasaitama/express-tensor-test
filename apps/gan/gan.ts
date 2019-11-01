import * as models from './model'

console.log(models.model);

class GAN<
      I extends {}
    , O 
  >
{
  public constructor(
    opts?:{
      generator:GANGenerator<I,O>,
      judger:GANJudger<I,O>
     }){
    //各モデルをロードする
    
  }

  private generator:GANGenerator<I,O>|undefined=undefined ;
  private judger:GANJudger<I,O>|undefined=undefined ;

  public requestGenerate(name:I):O|undefined{
    
    //リクエストタイプID:作成した内容　のデータを生成し、

    //判定士に 作成した内容　を渡し、リクエストタイプIDが一致するものを生成する
    return undefined;
  }
}

interface GG<I,O>{
  Generate(input:I):O;
}
interface GJ<I,O>{
  Judge(i:O):I;
}

abstract class GANGenerator<I,O> implements GG<I,O>{
  public abstract Generate(input:I):O;
  public constructor(){
  }
}

abstract class GANJudger<I,O> implements GJ<I,O>{
  public abstract Judge(i:O):I;
  public constructor(){
    
  }
}

class GANTargetLabelSet{

  private nameIndex:{
    [name:string]:number
  }={};
  private keyIndex:{
    [key:number]:string
  }={};


  public constructor(list:string[]){
    //転置インデックスを作成する
    list.forEach((k,v)=>{
      this.nameIndex[k]=v;
      this.keyIndex[v]=k;
      console.log([k,v]);
    });

    console.dir(this);
  }  

  public fromName( name:string):number
  {
    return 1;
  }
  
  public fromIndex(index:number):string{
    return "";
  }
}

new GANTargetLabelSet([
  "fff","ggg","aaa"
]);

enum TE{
  A="1",
  B="2",
  C="3"
}

//console.dir(TE);

const gan=new GAN<TE,number>();


