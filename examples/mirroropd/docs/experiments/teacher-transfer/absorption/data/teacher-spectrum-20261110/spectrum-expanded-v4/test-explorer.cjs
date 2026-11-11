const fs=require('fs'),vm=require('vm');
const elements={};
for(const id of ['feature','target','step','panel','student','direct','logx','logy','download','count','plot','rows']) elements[id]={value:'',checked:false,selectedOptions:[{text:'remaining'}],add(x){if(!this.value)this.value=x.value;}};
Object.assign(elements.target,{value:'remaining'});elements.step.value='64';elements.panel.value='All held-out';elements.student.value='all';
let calls=0;const context={document:{getElementById:id=>elements[id]},Option:function(text,value){return {text,value}},Plotly:{react(id,traces){calls++;if(!traces.length)throw Error('No traces');}},console};
vm.createContext(context);vm.runInContext(fs.readFileSync(__dirname+'/explorer-custom.js','utf8'),context);
const cases=[];for(const step of ['64','320'])for(const panel of ['All held-out','Math','Code','IF'])for(const target of ['remaining','final_rkl','late_relative_decrease']){elements.step.value=step;elements.panel.value=panel;elements.target.value=target;elements.target.onchange();cases.push({step,panel,target,count:elements.count.textContent});}
elements.direct.checked=true;elements.direct.onchange();
fs.writeFileSync(__dirname+'/explorer-runtime-check.json',JSON.stringify({syntax:'passed',mockDOM:'passed',plotly_render_calls:calls,cases,limitation:'No real-browser rendering or screenshot verification'},null,2));
console.log('passed',calls,'render paths');
