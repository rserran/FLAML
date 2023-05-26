"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[7641],{3905:(e,t,a)=>{a.d(t,{Zo:()=>p,kt:()=>c});var r=a(7294);function l(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function n(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,r)}return a}function o(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?n(Object(a),!0).forEach((function(t){l(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):n(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function i(e,t){if(null==e)return{};var a,r,l=function(e,t){if(null==e)return{};var a,r,l={},n=Object.keys(e);for(r=0;r<n.length;r++)a=n[r],t.indexOf(a)>=0||(l[a]=e[a]);return l}(e,t);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);for(r=0;r<n.length;r++)a=n[r],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(l[a]=e[a])}return l}var s=r.createContext({}),m=function(e){var t=r.useContext(s),a=t;return e&&(a="function"==typeof e?e(t):o(o({},t),e)),a},p=function(e){var t=m(e.components);return r.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var a=e.components,l=e.mdxType,n=e.originalType,s=e.parentName,p=i(e,["components","mdxType","originalType","parentName"]),d=m(a),c=l,_=d["".concat(s,".").concat(c)]||d[c]||u[c]||n;return a?r.createElement(_,o(o({ref:t},p),{},{components:a})):r.createElement(_,o({ref:t},p))}));function c(e,t){var a=arguments,l=t&&t.mdxType;if("string"==typeof e||l){var n=a.length,o=new Array(n);o[0]=d;var i={};for(var s in t)hasOwnProperty.call(t,s)&&(i[s]=t[s]);i.originalType=e,i.mdxType="string"==typeof e?e:l,o[1]=i;for(var m=2;m<n;m++)o[m]=a[m];return r.createElement.apply(null,o)}return r.createElement.apply(null,a)}d.displayName="MDXCreateElement"},6484:(e,t,a)=>{a.r(t),a.d(t,{contentTitle:()=>o,default:()=>p,frontMatter:()=>n,metadata:()=>i,toc:()=>s});var r=a(7462),l=(a(7294),a(3905));const n={},o="Default - Flamlized Estimator",i={unversionedId:"Examples/Default-Flamlized",id:"Examples/Default-Flamlized",isDocsHomePage:!1,title:"Default - Flamlized Estimator",description:'Flamlized estimators automatically use data-dependent default hyperparameter configurations for each estimator, offering a unique zero-shot AutoML capability, or "no tuning" AutoML.',source:"@site/docs/Examples/Default-Flamlized.md",sourceDirName:"Examples",slug:"/Examples/Default-Flamlized",permalink:"/FLAML/docs/Examples/Default-Flamlized",editUrl:"https://github.com/microsoft/FLAML/edit/main/website/docs/Examples/Default-Flamlized.md",tags:[],version:"current",frontMatter:{},sidebar:"docsSidebar",previous:{title:"AutoML for XGBoost",permalink:"/FLAML/docs/Examples/AutoML-for-XGBoost"},next:{title:"Integrate - AzureML",permalink:"/FLAML/docs/Examples/Integrate - AzureML"}},s=[{value:"Flamlized LGBMRegressor",id:"flamlized-lgbmregressor",children:[{value:"Prerequisites",id:"prerequisites",children:[],level:3},{value:"Zero-shot AutoML",id:"zero-shot-automl",children:[{value:"Sample output",id:"sample-output",children:[],level:4}],level:3},{value:"Suggest hyperparameters without training",id:"suggest-hyperparameters-without-training",children:[{value:"Sample output",id:"sample-output-1",children:[],level:4}],level:3}],level:2},{value:"Flamlized XGBClassifier",id:"flamlized-xgbclassifier",children:[{value:"Prerequisites",id:"prerequisites-1",children:[],level:3},{value:"Zero-shot AutoML",id:"zero-shot-automl-1",children:[{value:"Sample output",id:"sample-output-2",children:[],level:4}],level:3}],level:2}],m={toc:s};function p(e){let{components:t,...a}=e;return(0,l.kt)("wrapper",(0,r.Z)({},m,a,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"default---flamlized-estimator"},"Default - Flamlized Estimator"),(0,l.kt)("p",null,'Flamlized estimators automatically use data-dependent default hyperparameter configurations for each estimator, offering a unique zero-shot AutoML capability, or "no tuning" AutoML.'),(0,l.kt)("h2",{id:"flamlized-lgbmregressor"},"Flamlized LGBMRegressor"),(0,l.kt)("h3",{id:"prerequisites"},"Prerequisites"),(0,l.kt)("p",null,"This example requires the ","[autozero]"," option."),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-bash"},"pip install flaml[autozero] lightgbm openml\n")),(0,l.kt)("h3",{id:"zero-shot-automl"},"Zero-shot AutoML"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'from flaml.automl.data import load_openml_dataset\nfrom flaml.default import LGBMRegressor\nfrom flaml.automl.ml import sklearn_metric_loss_score\n\nX_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=537, data_dir="./")\nlgbm = LGBMRegressor()\nlgbm.fit(X_train, y_train)\ny_pred = lgbm.predict(X_test)\nprint("flamlized lgbm r2", "=", 1 - sklearn_metric_loss_score("r2", y_pred, y_test))\nprint(lgbm)\n')),(0,l.kt)("h4",{id:"sample-output"},"Sample output"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"load dataset from ./openml_ds537.pkl\nDataset name: houses\nX_train.shape: (15480, 8), y_train.shape: (15480,);\nX_test.shape: (5160, 8), y_test.shape: (5160,)\nflamlized lgbm r2 = 0.8537444671194614\nLGBMRegressor(colsample_bytree=0.7019911744574896,\n              learning_rate=0.022635758411078528, max_bin=511,\n              min_child_samples=2, n_estimators=4797, num_leaves=122,\n              reg_alpha=0.004252223402511765, reg_lambda=0.11288241427227624,\n              verbose=-1)\n")),(0,l.kt)("h3",{id:"suggest-hyperparameters-without-training"},"Suggest hyperparameters without training"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},'from flaml.data import load_openml_dataset\nfrom flaml.default import LGBMRegressor\nfrom flaml.ml import sklearn_metric_loss_score\n\nX_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=537, data_dir="./")\nlgbm = LGBMRegressor()\nhyperparams, estimator_name, X_transformed, y_transformed = lgbm.suggest_hyperparams(X_train, y_train)\nprint(hyperparams)\n')),(0,l.kt)("h4",{id:"sample-output-1"},"Sample output"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"load dataset from ./openml_ds537.pkl\nDataset name: houses\nX_train.shape: (15480, 8), y_train.shape: (15480,);\nX_test.shape: (5160, 8), y_test.shape: (5160,)\n{'n_estimators': 4797, 'num_leaves': 122, 'min_child_samples': 2, 'learning_rate': 0.022635758411078528, 'colsample_bytree': 0.7019911744574896, 'reg_alpha': 0.004252223402511765, 'reg_lambda': 0.11288241427227624, 'max_bin': 511, 'verbose': -1}\n")),(0,l.kt)("p",null,(0,l.kt)("a",{parentName:"p",href:"https://github.com/microsoft/FLAML/blob/main/notebook/zeroshot_lightgbm.ipynb"},"Link to notebook")," | ",(0,l.kt)("a",{parentName:"p",href:"https://colab.research.google.com/github/microsoft/FLAML/blob/main/notebook/zeroshot_lightgbm.ipynb"},"Open in colab")),(0,l.kt)("h2",{id:"flamlized-xgbclassifier"},"Flamlized XGBClassifier"),(0,l.kt)("h3",{id:"prerequisites-1"},"Prerequisites"),(0,l.kt)("p",null,"This example requires xgboost, sklearn, openml==0.10.2."),(0,l.kt)("h3",{id:"zero-shot-automl-1"},"Zero-shot AutoML"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'from flaml.automl.data import load_openml_dataset\nfrom flaml.default import XGBClassifier\nfrom flaml.automl.ml import sklearn_metric_loss_score\n\nX_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=1169, data_dir="./")\nxgb = XGBClassifier()\nxgb.fit(X_train, y_train)\ny_pred = xgb.predict(X_test)\nprint("flamlized xgb accuracy", "=", 1 - sklearn_metric_loss_score("accuracy", y_pred, y_test))\nprint(xgb)\n')),(0,l.kt)("h4",{id:"sample-output-2"},"Sample output"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"load dataset from ./openml_ds1169.pkl\nDataset name: airlines\nX_train.shape: (404537, 7), y_train.shape: (404537,);\nX_test.shape: (134846, 7), y_test.shape: (134846,)\nflamlized xgb accuracy = 0.6729009388487608\nXGBClassifier(base_score=0.5, booster='gbtree',\n              colsample_bylevel=0.4601573737792679, colsample_bynode=1,\n              colsample_bytree=1.0, gamma=0, gpu_id=-1, grow_policy='lossguide',\n              importance_type='gain', interaction_constraints='',\n              learning_rate=0.04039771837785377, max_delta_step=0, max_depth=0,\n              max_leaves=159, min_child_weight=0.3396294979905001, missing=nan,\n              monotone_constraints='()', n_estimators=540, n_jobs=4,\n              num_parallel_tree=1, random_state=0,\n              reg_alpha=0.0012362430984376035, reg_lambda=3.093428791531145,\n              scale_pos_weight=1, subsample=1.0, tree_method='hist',\n              use_label_encoder=False, validate_parameters=1, verbosity=0)\n")))}p.isMDXComponent=!0}}]);