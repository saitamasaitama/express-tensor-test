 function A(AP:number,Count:number,Spd:number,A:number,def:number){
  const D = AP + 100 - Count + Spd + A - def;
  console.log(D);
}

function B(AP:number,SPD:number,A:number,def:number){
  const D=  ( AP + 100 - 1 + 255 + A ) * 3 - def ;
  console.log(D);
}

A(10,100,100,10,200);
B(10,100,10,200);


