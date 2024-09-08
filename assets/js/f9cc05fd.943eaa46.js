"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[3165],{5680:(e,n,t)=>{t.d(n,{xA:()=>p,yg:()=>d});var a=t(6540);function l(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function r(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function i(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?r(Object(t),!0).forEach((function(n){l(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):r(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function o(e,n){if(null==e)return{};var t,a,l=function(e,n){if(null==e)return{};var t,a,l={},r=Object.keys(e);for(a=0;a<r.length;a++)t=r[a],n.indexOf(t)>=0||(l[t]=e[t]);return l}(e,n);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)t=r[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(l[t]=e[t])}return l}var s=a.createContext({}),c=function(e){var n=a.useContext(s),t=n;return e&&(t="function"==typeof e?e(n):i(i({},n),e)),t},p=function(e){var n=c(e.components);return a.createElement(s.Provider,{value:n},e.children)},m={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},u=a.forwardRef((function(e,n){var t=e.components,l=e.mdxType,r=e.originalType,s=e.parentName,p=o(e,["components","mdxType","originalType","parentName"]),u=c(t),d=l,g=u["".concat(s,".").concat(d)]||u[d]||m[d]||r;return t?a.createElement(g,i(i({ref:n},p),{},{components:t})):a.createElement(g,i({ref:n},p))}));function d(e,n){var t=arguments,l=n&&n.mdxType;if("string"==typeof e||l){var r=t.length,i=new Array(r);i[0]=u;var o={};for(var s in n)hasOwnProperty.call(n,s)&&(o[s]=n[s]);o.originalType=e,o.mdxType="string"==typeof e?e:l,i[1]=o;for(var c=2;c<r;c++)i[c]=t[c];return a.createElement.apply(null,i)}return a.createElement.apply(null,t)}u.displayName="MDXCreateElement"},7439:(e,n,t)=>{t.r(n),t.d(n,{contentTitle:()=>i,default:()=>p,frontMatter:()=>r,metadata:()=>o,toc:()=>s});var a=t(8168),l=(t(6540),t(5680));const r={sidebar_label:"autovw",title:"onlineml.autovw"},i=void 0,o={unversionedId:"reference/onlineml/autovw",id:"reference/onlineml/autovw",isDocsHomePage:!1,title:"onlineml.autovw",description:"AutoVW Objects",source:"@site/docs/reference/onlineml/autovw.md",sourceDirName:"reference/onlineml",slug:"/reference/onlineml/autovw",permalink:"/FLAML/docs/reference/onlineml/autovw",editUrl:"https://github.com/microsoft/FLAML/edit/main/website/docs/reference/onlineml/autovw.md",tags:[],version:"current",frontMatter:{sidebar_label:"autovw",title:"onlineml.autovw"},sidebar:"referenceSideBar",previous:{title:"mlflow",permalink:"/FLAML/docs/reference/fabric/mlflow"},next:{title:"trial",permalink:"/FLAML/docs/reference/onlineml/trial"}},s=[{value:"AutoVW Objects",id:"autovw-objects",children:[{value:"__init__",id:"__init__",children:[],level:4},{value:"predict",id:"predict",children:[],level:4},{value:"learn",id:"learn",children:[],level:4},{value:"get_ns_feature_dim_from_vw_example",id:"get_ns_feature_dim_from_vw_example",children:[],level:4}],level:2}],c={toc:s};function p(e){let{components:n,...t}=e;return(0,l.yg)("wrapper",(0,a.A)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,l.yg)("h2",{id:"autovw-objects"},"AutoVW Objects"),(0,l.yg)("pre",null,(0,l.yg)("code",{parentName:"pre",className:"language-python"},"class AutoVW()\n")),(0,l.yg)("p",null,"Class for the AutoVW algorithm."),(0,l.yg)("h4",{id:"__init__"},"_","_","init","_","_"),(0,l.yg)("pre",null,(0,l.yg)("code",{parentName:"pre",className:"language-python"},'def __init__(max_live_model_num: int, search_space: dict, init_config: Optional[dict] = {}, min_resource_lease: Optional[Union[str, float]] = "auto", automl_runner_args: Optional[dict] = {}, scheduler_args: Optional[dict] = {}, model_select_policy: Optional[str] = "threshold_loss_ucb", metric: Optional[str] = "mae_clipped", random_seed: Optional[int] = None, model_selection_mode: Optional[str] = "min", cb_coef: Optional[float] = None)\n')),(0,l.yg)("p",null,"Constructor."),(0,l.yg)("p",null,(0,l.yg)("strong",{parentName:"p"},"Arguments"),":"),(0,l.yg)("ul",null,(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"max_live_model_num")," - An int to specify the maximum number of\n'live' models, which, in other words, is the maximum number\nof models allowed to update in each learning iteraction."),(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"search_space")," - A dictionary of the search space. This search space\nincludes both hyperparameters we want to tune and fixed\nhyperparameters. In the latter case, the value is a fixed value."),(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"init_config")," - A dictionary of a partial or full initial config,\ne.g. {'interactions': set(), 'learning_rate': 0.5}"),(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"min_resource_lease")," - string or float | The minimum resource lease\nassigned to a particular model/trial. If set as 'auto', it will\nbe calculated automatically."),(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"automl_runner_args")," - A dictionary of configuration for the OnlineTrialRunner.\nIf set {}, default values will be used, which is equivalent to using\nthe following configs.")),(0,l.yg)("p",null,(0,l.yg)("strong",{parentName:"p"},"Example"),":"),(0,l.yg)("pre",null,(0,l.yg)("code",{parentName:"pre",className:"language-python"},'automl_runner_args = {\n    "champion_test_policy": \'loss_ucb\', # the statistic test for a better champion\n    "remove_worse": False,              # whether to do worse than test\n}\n')),(0,l.yg)("ul",null,(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"scheduler_args")," - A dictionary of configuration for the scheduler.\nIf set {}, default values will be used, which is equivalent to using the\nfollowing config.")),(0,l.yg)("p",null,(0,l.yg)("strong",{parentName:"p"},"Example"),":"),(0,l.yg)("pre",null,(0,l.yg)("code",{parentName:"pre",className:"language-python"},'scheduler_args = {\n    "keep_challenger_metric": \'ucb\',  # what metric to use when deciding the top performing challengers\n    "keep_challenger_ratio": 0.5,     # denotes the ratio of top performing challengers to keep live\n    "keep_champion": True,            # specifcies whether to keep the champion always running\n}\n')),(0,l.yg)("ul",null,(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"model_select_policy")," - A string in ","['threshold_loss_ucb',\n'threshold_loss_lcb', 'threshold_loss_avg', 'loss_ucb', 'loss_lcb',\n'loss_avg']"," to specify how to select one model to do prediction from\nthe live model pool. Default value is 'threshold_loss_ucb'."),(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"metric")," - A string in ","['mae_clipped', 'mae', 'mse', 'absolute_clipped',\n'absolute', 'squared']"," to specify the name of the loss function used\nfor calculating the progressive validation loss in ChaCha."),(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"random_seed")," - An integer of the random seed used in the searcher\n(more specifically this the random seed for ConfigOracle)."),(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"model_selection_mode")," - A string in ","['min', 'max']"," to specify the objective as\nminimization or maximization."),(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"cb_coef")," - A float coefficient (optional) used in the sample complexity bound.")),(0,l.yg)("h4",{id:"predict"},"predict"),(0,l.yg)("pre",null,(0,l.yg)("code",{parentName:"pre",className:"language-python"},"def predict(data_sample)\n")),(0,l.yg)("p",null,"Predict on the input data sample."),(0,l.yg)("p",null,(0,l.yg)("strong",{parentName:"p"},"Arguments"),":"),(0,l.yg)("ul",null,(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"data_sample")," - one data example in vw format.")),(0,l.yg)("h4",{id:"learn"},"learn"),(0,l.yg)("pre",null,(0,l.yg)("code",{parentName:"pre",className:"language-python"},"def learn(data_sample)\n")),(0,l.yg)("p",null,"Perform one online learning step with the given data sample."),(0,l.yg)("p",null,(0,l.yg)("strong",{parentName:"p"},"Arguments"),":"),(0,l.yg)("ul",null,(0,l.yg)("li",{parentName:"ul"},(0,l.yg)("inlineCode",{parentName:"li"},"data_sample")," - one data example in vw format. It will be used to\nupdate the vw model.")),(0,l.yg)("h4",{id:"get_ns_feature_dim_from_vw_example"},"get","_","ns","_","feature","_","dim","_","from","_","vw","_","example"),(0,l.yg)("pre",null,(0,l.yg)("code",{parentName:"pre",className:"language-python"},"@staticmethod\ndef get_ns_feature_dim_from_vw_example(vw_example) -> dict\n")),(0,l.yg)("p",null,"Get a dictionary of feature dimensionality for each namespace singleton."))}p.isMDXComponent=!0}}]);