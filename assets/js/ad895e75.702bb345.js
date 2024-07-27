"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[4288],{3905:(e,t,a)=>{a.d(t,{Zo:()=>m,kt:()=>d});var n=a(7294);function o(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function r(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function i(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?r(Object(a),!0).forEach((function(t){o(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function s(e,t){if(null==e)return{};var a,n,o=function(e,t){if(null==e)return{};var a,n,o={},r=Object.keys(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||(o[a]=e[a]);return o}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(o[a]=e[a])}return o}var l=n.createContext({}),p=function(e){var t=n.useContext(l),a=t;return e&&(a="function"==typeof e?e(t):i(i({},t),e)),a},m=function(e){var t=p(e.components);return n.createElement(l.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},c=n.forwardRef((function(e,t){var a=e.components,o=e.mdxType,r=e.originalType,l=e.parentName,m=s(e,["components","mdxType","originalType","parentName"]),c=p(a),d=o,h=c["".concat(l,".").concat(d)]||c[d]||u[d]||r;return a?n.createElement(h,i(i({ref:t},m),{},{components:a})):n.createElement(h,i({ref:t},m))}));function d(e,t){var a=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var r=a.length,i=new Array(r);i[0]=c;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s.mdxType="string"==typeof e?e:o,i[1]=s;for(var p=2;p<r;p++)i[p]=a[p];return n.createElement.apply(null,i)}return n.createElement.apply(null,a)}c.displayName="MDXCreateElement"},3581:(e,t,a)=>{a.r(t),a.d(t,{contentTitle:()=>i,default:()=>m,frontMatter:()=>r,metadata:()=>s,toc:()=>l});var n=a(7462),o=(a(7294),a(3905));const r={},i="Frequently Asked Questions",s={unversionedId:"FAQ",id:"FAQ",isDocsHomePage:!1,title:"Frequently Asked Questions",description:"Guidelines on how to set a hyperparameter search space",source:"@site/docs/FAQ.md",sourceDirName:".",slug:"/FAQ",permalink:"/FLAML/docs/FAQ",editUrl:"https://github.com/microsoft/FLAML/edit/main/website/docs/FAQ.md",tags:[],version:"current",frontMatter:{}},l=[{value:"Guidelines on how to set a hyperparameter search space",id:"guidelines-on-how-to-set-a-hyperparameter-search-space",children:[],level:3},{value:"Guidelines on parallel vs seqential tuning",id:"guidelines-on-parallel-vs-seqential-tuning",children:[],level:3},{value:"Guidelines on creating and tuning a custom estimator",id:"guidelines-on-creating-and-tuning-a-custom-estimator",children:[],level:3},{value:"About <code>low_cost_partial_config</code> in <code>tune</code>.",id:"about-low_cost_partial_config-in-tune",children:[],level:3},{value:"How does FLAML handle imbalanced data (unequal distribution of target classes in classification task)?",id:"how-does-flaml-handle-imbalanced-data-unequal-distribution-of-target-classes-in-classification-task",children:[],level:3},{value:"How to interpret model performance? Is it possible for me to visualize feature importance, SHAP values, optimization history?",id:"how-to-interpret-model-performance-is-it-possible-for-me-to-visualize-feature-importance-shap-values-optimization-history",children:[],level:3},{value:"How to resolve out-of-memory error in <code>AutoML.fit()</code>",id:"how-to-resolve-out-of-memory-error-in-automlfit",children:[],level:3},{value:"How to get the best config of an estimator and use it to train the original model outside FLAML?",id:"how-to-get-the-best-config-of-an-estimator-and-use-it-to-train-the-original-model-outside-flaml",children:[],level:3}],p={toc:l};function m(e){let{components:t,...a}=e;return(0,o.kt)("wrapper",(0,n.Z)({},p,a,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"frequently-asked-questions"},"Frequently Asked Questions"),(0,o.kt)("h3",{id:"guidelines-on-how-to-set-a-hyperparameter-search-space"},(0,o.kt)("a",{parentName:"h3",href:"Use-Cases/Tune-User-Defined-Function#details-and-guidelines-on-hyperparameter-search-space"},"Guidelines on how to set a hyperparameter search space")),(0,o.kt)("h3",{id:"guidelines-on-parallel-vs-seqential-tuning"},(0,o.kt)("a",{parentName:"h3",href:"Use-Cases/Task-Oriented-AutoML#guidelines-on-parallel-vs-sequential-tuning"},"Guidelines on parallel vs seqential tuning")),(0,o.kt)("h3",{id:"guidelines-on-creating-and-tuning-a-custom-estimator"},(0,o.kt)("a",{parentName:"h3",href:"Use-Cases/Task-Oriented-AutoML#guidelines-on-tuning-a-custom-estimator"},"Guidelines on creating and tuning a custom estimator")),(0,o.kt)("h3",{id:"about-low_cost_partial_config-in-tune"},"About ",(0,o.kt)("inlineCode",{parentName:"h3"},"low_cost_partial_config")," in ",(0,o.kt)("inlineCode",{parentName:"h3"},"tune"),"."),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("p",{parentName:"li"},"Definition and purpose: The ",(0,o.kt)("inlineCode",{parentName:"p"},"low_cost_partial_config")," is a dictionary of subset of the hyperparameter coordinates whose value corresponds to a configuration with known low-cost (i.e., low computation cost for training the corresponding model).  The concept of low/high-cost is meaningful in the case where a subset of the hyperparameters to tune directly affects the computation cost for training the model. For example, ",(0,o.kt)("inlineCode",{parentName:"p"},"n_estimators")," and ",(0,o.kt)("inlineCode",{parentName:"p"},"max_leaves")," are known to affect the training cost of tree-based learners. We call this subset of hyperparameters, ",(0,o.kt)("em",{parentName:"p"},"cost-related hyperparameters"),". In such scenarios, if you are aware of low-cost configurations for the cost-related hyperparameters, you are recommended to set them as the ",(0,o.kt)("inlineCode",{parentName:"p"},"low_cost_partial_config"),". Using the tree-based method example again, since we know that small ",(0,o.kt)("inlineCode",{parentName:"p"},"n_estimators")," and  ",(0,o.kt)("inlineCode",{parentName:"p"},"max_leaves")," generally correspond to simpler models and thus lower cost, we set ",(0,o.kt)("inlineCode",{parentName:"p"},"{'n_estimators': 4, 'max_leaves': 4}")," as the ",(0,o.kt)("inlineCode",{parentName:"p"},"low_cost_partial_config")," by default (note that ",(0,o.kt)("inlineCode",{parentName:"p"},"4")," is the lower bound of search space for these two hyperparameters), e.g., in ",(0,o.kt)("a",{parentName:"p",href:"https://github.com/microsoft/FLAML/blob/main/flaml/model.py#L215"},"LGBM"),".  Configuring ",(0,o.kt)("inlineCode",{parentName:"p"},"low_cost_partial_config")," helps the search algorithms make more cost-efficient choices.\nIn AutoML, the ",(0,o.kt)("inlineCode",{parentName:"p"},"low_cost_init_value")," in ",(0,o.kt)("inlineCode",{parentName:"p"},"search_space()")," function for each estimator serves the same role.")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("p",{parentName:"li"},"Usage in practice: It is recommended to configure it if there are cost-related hyperparameters in your tuning task and you happen to know the low-cost values for them, but it is not required (It is fine to leave it the default value, i.e., ",(0,o.kt)("inlineCode",{parentName:"p"},"None"),").")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("p",{parentName:"li"},"How does it work: ",(0,o.kt)("inlineCode",{parentName:"p"},"low_cost_partial_config")," if configured, will be used as an initial point of the search. It also affects the search trajectory. For more details about how does it play a role in the search algorithms, please refer to the papers about the search algorithms used: Section 2 of ",(0,o.kt)("a",{parentName:"p",href:"https://arxiv.org/pdf/2005.01571.pdf"},"Frugal Optimization for Cost-related Hyperparameters (CFO)")," and Section 3 of ",(0,o.kt)("a",{parentName:"p",href:"https://openreview.net/pdf?id=VbLH04pRA3"},"Economical Hyperparameter Optimization with Blended Search Strategy (BlendSearch)"),"."))),(0,o.kt)("h3",{id:"how-does-flaml-handle-imbalanced-data-unequal-distribution-of-target-classes-in-classification-task"},"How does FLAML handle imbalanced data (unequal distribution of target classes in classification task)?"),(0,o.kt)("p",null,"Currently FLAML does several things for imbalanced data."),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"When a class contains fewer than 20 examples, we repeatedly add these examples to the training data until the count is at least 20."),(0,o.kt)("li",{parentName:"ol"},"We use stratified sampling when doing holdout and kf."),(0,o.kt)("li",{parentName:"ol"},"We make sure no class is empty in both training and holdout data."),(0,o.kt)("li",{parentName:"ol"},"We allow users to pass ",(0,o.kt)("inlineCode",{parentName:"li"},"sample_weight")," to ",(0,o.kt)("inlineCode",{parentName:"li"},"AutoML.fit()"),"."),(0,o.kt)("li",{parentName:"ol"},"User can customize the weight of each class by setting the ",(0,o.kt)("inlineCode",{parentName:"li"},"custom_hp")," or ",(0,o.kt)("inlineCode",{parentName:"li"},"fit_kwargs_by_estimator")," arguments. For example, the following code sets the weight for pos vs. neg as 2:1 for the RandomForest estimator:")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'from flaml import AutoML\nfrom sklearn.datasets import load_iris\n\nX_train, y_train = load_iris(return_X_y=True)\nautoml = AutoML()\nautoml_settings = {\n    "time_budget": 2,\n    "task": "classification",\n    "log_file_name": "test/iris.log",\n    "estimator_list": ["rf", "xgboost"],\n}\n\nautoml_settings["custom_hp"] = {\n    "xgboost": {\n        "scale_pos_weight": {\n            "domain": 0.5,\n            "init_value": 0.5,\n        }\n    },\n    "rf": {"class_weight": {"domain": "balanced", "init_value": "balanced"}},\n}\nprint(automl.model)\n')),(0,o.kt)("h3",{id:"how-to-interpret-model-performance-is-it-possible-for-me-to-visualize-feature-importance-shap-values-optimization-history"},"How to interpret model performance? Is it possible for me to visualize feature importance, SHAP values, optimization history?"),(0,o.kt)("p",null,"You can use ",(0,o.kt)("inlineCode",{parentName:"p"},"automl.model.estimator.feature_importances_")," to get the ",(0,o.kt)("inlineCode",{parentName:"p"},"feature_importances_")," for the best model found by automl. See an ",(0,o.kt)("a",{parentName:"p",href:"Examples/AutoML-for-XGBoost#plot-feature-importance"},"example"),"."),(0,o.kt)("p",null,"Packages such as ",(0,o.kt)("inlineCode",{parentName:"p"},"azureml-interpret")," and ",(0,o.kt)("inlineCode",{parentName:"p"},"sklearn.inspection.permutation_importance")," can be used on ",(0,o.kt)("inlineCode",{parentName:"p"},"automl.model.estimator")," to explain the selected model.\nModel explanation is frequently asked and adding a native support may be a good feature. Suggestions/contributions are welcome."),(0,o.kt)("p",null,"Optimization history can be checked from the ",(0,o.kt)("a",{parentName:"p",href:"Use-Cases/Task-Oriented-AutoML#log-the-trials"},"log"),". You can also ",(0,o.kt)("a",{parentName:"p",href:"Use-Cases/Task-Oriented-AutoML#plot-learning-curve"},"retrieve the log and plot the learning curve"),"."),(0,o.kt)("h3",{id:"how-to-resolve-out-of-memory-error-in-automlfit"},"How to resolve out-of-memory error in ",(0,o.kt)("inlineCode",{parentName:"h3"},"AutoML.fit()")),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Set ",(0,o.kt)("inlineCode",{parentName:"li"},"free_mem_ratio")," a float between 0 and 1. For example, 0.2 means try to keep free memory above 20% of total memory. Training may be early stopped for memory consumption reason when this is set."),(0,o.kt)("li",{parentName:"ul"},"Set ",(0,o.kt)("inlineCode",{parentName:"li"},"model_history")," False."),(0,o.kt)("li",{parentName:"ul"},"If your data are already preprocessed, set ",(0,o.kt)("inlineCode",{parentName:"li"},"skip_transform")," False. If you can preprocess the data before the fit starts, this setting can save memory needed for preprocessing in ",(0,o.kt)("inlineCode",{parentName:"li"},"fit"),"."),(0,o.kt)("li",{parentName:"ul"},"If the OOM error only happens for some particular trials:",(0,o.kt)("ul",{parentName:"li"},(0,o.kt)("li",{parentName:"ul"},"set ",(0,o.kt)("inlineCode",{parentName:"li"},"use_ray")," True. This will increase the overhead per trial but can keep the AutoML process running when a single trial fails due to OOM error."),(0,o.kt)("li",{parentName:"ul"},"provide a more accurate ",(0,o.kt)("a",{parentName:"li",href:"reference/automl/model#size"},(0,o.kt)("inlineCode",{parentName:"a"},"size"))," function for the memory bytes consumption of each config for the estimator causing this error."),(0,o.kt)("li",{parentName:"ul"},"modify the ",(0,o.kt)("a",{parentName:"li",href:"Use-Cases/Task-Oriented-AutoML#a-shortcut-to-override-the-search-space"},"search space")," for the estimators causing this error."),(0,o.kt)("li",{parentName:"ul"},"or remove this estimator from the ",(0,o.kt)("inlineCode",{parentName:"li"},"estimator_list"),"."))),(0,o.kt)("li",{parentName:"ul"},"If the OOM error happens when ensembling, consider disabling ensemble, or use a cheaper ensemble option. (",(0,o.kt)("a",{parentName:"li",href:"Use-Cases/Task-Oriented-AutoML#ensemble"},"Example"),").")),(0,o.kt)("h3",{id:"how-to-get-the-best-config-of-an-estimator-and-use-it-to-train-the-original-model-outside-flaml"},"How to get the best config of an estimator and use it to train the original model outside FLAML?"),(0,o.kt)("p",null,"When you finished training an AutoML estimator, you may want to use it in other code w/o depending on FLAML. You can get the ",(0,o.kt)("inlineCode",{parentName:"p"},"automl.best_config")," and convert it to the parameters of the original model with below code:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'from flaml import AutoML\nfrom sklearn.datasets import load_iris\n\nX, y = load_iris(return_X_y=True)\n\nautoml = AutoML(settings={"time_budget": 3})\nautoml.fit(X, y)\n\nprint(f"{automl.best_estimator=}")\nprint(f"{automl.best_config=}")\nprint(f"params for best estimator: {automl.model.config2params(automl.best_config)}")\n')))}m.isMDXComponent=!0}}]);