"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[4136],{3905:(e,t,n)=>{n.d(t,{Zo:()=>c,kt:()=>d});var a=n(7294);function l(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){l(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function r(e,t){if(null==e)return{};var n,a,l=function(e,t){if(null==e)return{};var n,a,l={},o=Object.keys(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||(l[n]=e[n]);return l}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(l[n]=e[n])}return l}var s=a.createContext({}),p=function(e){var t=a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},c=function(e){var t=p(e.components);return a.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},m=a.forwardRef((function(e,t){var n=e.components,l=e.mdxType,o=e.originalType,s=e.parentName,c=r(e,["components","mdxType","originalType","parentName"]),m=p(n),d=l,h=m["".concat(s,".").concat(d)]||m[d]||u[d]||o;return n?a.createElement(h,i(i({ref:t},c),{},{components:n})):a.createElement(h,i({ref:t},c))}));function d(e,t){var n=arguments,l=t&&t.mdxType;if("string"==typeof e||l){var o=n.length,i=new Array(o);i[0]=m;var r={};for(var s in t)hasOwnProperty.call(t,s)&&(r[s]=t[s]);r.originalType=e,r.mdxType="string"==typeof e?e:l,i[1]=r;for(var p=2;p<o;p++)i[p]=n[p];return a.createElement.apply(null,i)}return a.createElement.apply(null,n)}m.displayName="MDXCreateElement"},7097:(e,t,n)=>{n.r(t),n.d(t,{contentTitle:()=>i,default:()=>c,frontMatter:()=>o,metadata:()=>r,toc:()=>s});var a=n(7462),l=(n(7294),n(3905));const o={sidebar_label:"completion",title:"autogen.oai.completion"},i=void 0,r={unversionedId:"reference/autogen/oai/completion",id:"reference/autogen/oai/completion",isDocsHomePage:!1,title:"autogen.oai.completion",description:"Completion Objects",source:"@site/docs/reference/autogen/oai/completion.md",sourceDirName:"reference/autogen/oai",slug:"/reference/autogen/oai/completion",permalink:"/FLAML/docs/reference/autogen/oai/completion",editUrl:"https://github.com/microsoft/FLAML/edit/main/website/docs/reference/autogen/oai/completion.md",tags:[],version:"current",frontMatter:{sidebar_label:"completion",title:"autogen.oai.completion"},sidebar:"referenceSideBar",previous:{title:"user_proxy_agent",permalink:"/FLAML/docs/reference/autogen/agent/user_proxy_agent"},next:{title:"openai_utils",permalink:"/FLAML/docs/reference/autogen/oai/openai_utils"}},s=[{value:"Completion Objects",id:"completion-objects",children:[{value:"set_cache",id:"set_cache",children:[],level:4},{value:"clear_cache",id:"clear_cache",children:[],level:4},{value:"tune",id:"tune",children:[],level:4},{value:"create",id:"create",children:[],level:4},{value:"test",id:"test",children:[],level:4},{value:"cost",id:"cost",children:[],level:4},{value:"extract_text",id:"extract_text",children:[],level:4},{value:"extract_text_or_function_call",id:"extract_text_or_function_call",children:[],level:4},{value:"logged_history",id:"logged_history",children:[],level:4},{value:"start_logging",id:"start_logging",children:[],level:4},{value:"stop_logging",id:"stop_logging",children:[],level:4}],level:2},{value:"ChatCompletion Objects",id:"chatcompletion-objects",children:[],level:2}],p={toc:s};function c(e){let{components:t,...n}=e;return(0,l.kt)("wrapper",(0,a.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h2",{id:"completion-objects"},"Completion Objects"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"class Completion(openai_Completion)\n")),(0,l.kt)("p",null,"A class for OpenAI completion API."),(0,l.kt)("p",null,"It also supports: ChatCompletion, Azure OpenAI API."),(0,l.kt)("h4",{id:"set_cache"},"set","_","cache"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'@classmethod\ndef set_cache(cls, seed: Optional[int] = 41, cache_path_root: Optional[str] = ".cache")\n')),(0,l.kt)("p",null,"Set cache path."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"seed")," ",(0,l.kt)("em",{parentName:"li"},"int, Optional")," - The integer identifier for the pseudo seed.\nResults corresponding to different seeds will be cached in different places."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"cache_path")," ",(0,l.kt)("em",{parentName:"li"},"str, Optional")," - The root path for the cache.\nThe complete cache path will be {cache_path}/{seed}.")),(0,l.kt)("h4",{id:"clear_cache"},"clear","_","cache"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'@classmethod\ndef clear_cache(cls, seed: Optional[int] = None, cache_path_root: Optional[str] = ".cache")\n')),(0,l.kt)("p",null,"Clear cache."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"seed")," ",(0,l.kt)("em",{parentName:"li"},"int, Optional")," - The integer identifier for the pseudo seed.\nIf omitted, all caches under cache_path_root will be cleared."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"cache_path")," ",(0,l.kt)("em",{parentName:"li"},"str, Optional")," - The root path for the cache.\nThe complete cache path will be {cache_path}/{seed}.")),(0,l.kt)("h4",{id:"tune"},"tune"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"@classmethod\ndef tune(cls, data: List[Dict], metric: str, mode: str, eval_func: Callable, log_file_name: Optional[str] = None, inference_budget: Optional[float] = None, optimization_budget: Optional[float] = None, num_samples: Optional[int] = 1, logging_level: Optional[int] = logging.WARNING, **config, ,)\n")),(0,l.kt)("p",null,"Tune the parameters for the OpenAI API call."),(0,l.kt)("p",null,"TODO: support parallel tuning with ray or spark.\nTODO: support agg_method as in test"),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"data")," ",(0,l.kt)("em",{parentName:"li"},"list")," - The list of data points."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"metric")," ",(0,l.kt)("em",{parentName:"li"},"str")," - The metric to optimize."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"mode")," ",(0,l.kt)("em",{parentName:"li"},"str"),' - The optimization mode, "min" or "max.'),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"eval_func")," ",(0,l.kt)("em",{parentName:"li"},"Callable")," - The evaluation function for responses.\nThe function should take a list of responses and a data point as input,\nand return a dict of metrics. For example,")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'def eval_func(responses, **data):\n    solution = data["solution"]\n    success_list = []\n    n = len(responses)\n    for i in range(n):\n        response = responses[i]\n        succeed = is_equiv_chain_of_thought(response, solution)\n        success_list.append(succeed)\n    return {\n        "expected_success": 1 - pow(1 - sum(success_list) / n, n),\n        "success": any(s for s in success_list),\n    }\n')),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"log_file_name")," ",(0,l.kt)("em",{parentName:"li"},"str, optional")," - The log file."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"inference_budget")," ",(0,l.kt)("em",{parentName:"li"},"float, optional")," - The inference budget, dollar per instance."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"optimization_budget")," ",(0,l.kt)("em",{parentName:"li"},"float, optional")," - The optimization budget, dollar in total."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"num_samples")," ",(0,l.kt)("em",{parentName:"li"},"int, optional")," - The number of samples to evaluate.\n-1 means no hard restriction in the number of trials\nand the actual number is decided by optimization_budget. Defaults to 1."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"logging_level")," ",(0,l.kt)("em",{parentName:"li"},"optional")," - logging level. Defaults to logging.WARNING."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"**config")," ",(0,l.kt)("em",{parentName:"li"},"dict")," - The search space to update over the default search.\nFor prompt, please provide a string/Callable or a list of strings/Callables.",(0,l.kt)("ul",{parentName:"li"},(0,l.kt)("li",{parentName:"ul"},'If prompt is provided for chat models, it will be converted to messages under role "user".'),(0,l.kt)("li",{parentName:"ul"},"Do not provide both prompt and messages for chat models, but provide either of them."),(0,l.kt)("li",{parentName:"ul"},"A string template will be used to generate a prompt for each data instance\nusing ",(0,l.kt)("inlineCode",{parentName:"li"},"prompt.format(**data)"),"."),(0,l.kt)("li",{parentName:"ul"},"A callable template will be used to generate a prompt for each data instance\nusing ",(0,l.kt)("inlineCode",{parentName:"li"},"prompt(data)"),'.\nFor stop, please provide a string, a list of strings, or a list of lists of strings.\nFor messages (chat models only), please provide a list of messages (for a single chat prefix)\nor a list of lists of messages (for multiple choices of chat prefix to choose from).\nEach message should be a dict with keys "role" and "content". The value of "content" can be a string/Callable template.')))),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"dict")," - The optimized hyperparameter setting."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"tune.ExperimentAnalysis")," - The tuning results.")),(0,l.kt)("h4",{id:"create"},"create"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"@classmethod\ndef create(cls, context: Optional[Dict] = None, use_cache: Optional[bool] = True, config_list: Optional[List[Dict]] = None, filter_func: Optional[Callable[[Dict, Dict, Dict], bool]] = None, raise_on_ratelimit_or_timeout: Optional[bool] = True, **config, ,)\n")),(0,l.kt)("p",null,"Make a completion for a given context."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"context")," ",(0,l.kt)("em",{parentName:"li"},"Dict, Optional")," - The context to instantiate the prompt.\nIt needs to contain keys that are used by the prompt template or the filter function.\nE.g., ",(0,l.kt)("inlineCode",{parentName:"li"},'prompt="Complete the following sentence: {prefix}, context={"prefix": "Today I feel"}'),'.\nThe actual prompt will be:\n"Complete the following sentence: Today I feel".\nMore examples can be found at ',(0,l.kt)("a",{parentName:"li",href:"/docs/Use-Cases/Auto-Generation#templating"},"templating"),"."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"use_cache")," ",(0,l.kt)("em",{parentName:"li"},"bool, Optional")," - Whether to use cached responses."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"config_list")," ",(0,l.kt)("em",{parentName:"li"},"List, Optional")," - List of configurations for the completion to try.\nThe first one that does not raise an error will be used.\nOnly the differences from the default config need to be provided.\nE.g.,")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'response = oai.Completion.create(\n    config_list=[\n        {\n            "model": "gpt-4",\n            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),\n            "api_type": "azure",\n            "api_base": os.environ.get("AZURE_OPENAI_API_BASE"),\n            "api_version": "2023-03-15-preview",\n        },\n        {\n            "model": "gpt-3.5-turbo",\n            "api_key": os.environ.get("OPENAI_API_KEY"),\n            "api_type": "open_ai",\n            "api_base": "https://api.openai.com/v1",\n        },\n        {\n            "model": "llama-7B",\n            "api_base": "http://127.0.0.1:8080",\n            "api_type": "open_ai",\n        }\n    ],\n    prompt="Hi",\n)\n')),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"filter_func")," ",(0,l.kt)("em",{parentName:"li"},"Callable, Optional")," - A function that takes in the context, the config and the response and returns a boolean to indicate whether the response is valid. E.g.,")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'def yes_or_no_filter(context, config, response):\n    return context.get("yes_or_no_choice", False) is False or any(\n        text in ["Yes.", "No."] for text in oai.Completion.extract_text(response)\n    )\n')),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"raise_on_ratelimit_or_timeout")," ",(0,l.kt)("em",{parentName:"li"},"bool, Optional")," - Whether to raise RateLimitError or Timeout when all configs fail.\nWhen set to False, -1 will be returned when all configs fail."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"**config"),' - Configuration for the openai API call. This is used as parameters for calling openai API.\nBesides the parameters for the openai API call, it can also contain a seed (int) for the cache.\nThis is useful when implementing "controlled randomness" for the completion.\nAlso, the "prompt" or "messages" parameter can contain a template (str or Callable) which will be instantiated with the context.')),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("p",null,"  Responses from OpenAI API."),(0,l.kt)("h4",{id:"test"},"test"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'@classmethod\ndef test(cls, data, eval_func=None, use_cache=True, agg_method="avg", return_responses_and_per_instance_result=False, logging_level=logging.WARNING, **config, ,)\n')),(0,l.kt)("p",null,"Evaluate the responses created with the config for the OpenAI API call."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"data")," ",(0,l.kt)("em",{parentName:"li"},"list")," - The list of test data points."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"eval_func")," ",(0,l.kt)("em",{parentName:"li"},"Callable")," - The evaluation function for responses per data instance.\nThe function should take a list of responses and a data point as input,\nand return a dict of metrics. You need to either provide a valid callable\neval_func; or do not provide one (set None) but call the test function after\ncalling the tune function in which a eval_func is provided.\nIn the latter case we will use the eval_func provided via tune function.\nDefaults to None.")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'def eval_func(responses, **data):\n    solution = data["solution"]\n    success_list = []\n    n = len(responses)\n    for i in range(n):\n        response = responses[i]\n        succeed = is_equiv_chain_of_thought(response, solution)\n        success_list.append(succeed)\n    return {\n        "expected_success": 1 - pow(1 - sum(success_list) / n, n),\n        "success": any(s for s in success_list),\n    }\n')),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"use_cache")," ",(0,l.kt)("em",{parentName:"li"},"bool, Optional")," - Whether to use cached responses. Defaults to True."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"agg_method")," ",(0,l.kt)("em",{parentName:"li"},"str, Callable or a dict of Callable")," - Result aggregation method (across\nmultiple instances) for each of the metrics. Defaults to 'avg'.\nAn example agg_method in str:")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"agg_method = 'median'\n")),(0,l.kt)("p",null,"  An example agg_method in a Callable:"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"agg_method = np.median\n")),(0,l.kt)("p",null,"  An example agg_method in a dict of Callable:"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"agg_method={'median_success': np.median, 'avg_success': np.mean}\n")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"return_responses_and_per_instance_result")," ",(0,l.kt)("em",{parentName:"li"},"bool")," - Whether to also return responses\nand per instance results in addition to the aggregated results."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"logging_level")," ",(0,l.kt)("em",{parentName:"li"},"optional")," - logging level. Defaults to logging.WARNING."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"**config")," ",(0,l.kt)("em",{parentName:"li"},"dict")," - parametes passed to the openai api call ",(0,l.kt)("inlineCode",{parentName:"li"},"create()"),".")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("p",null,"  None when no valid eval_func is provided in either test or tune;\nOtherwise, a dict of aggregated results, responses and per instance results if ",(0,l.kt)("inlineCode",{parentName:"p"},"return_responses_and_per_instance_result")," is True;\nOtherwise, a dict of aggregated results (responses and per instance results are not returned)."),(0,l.kt)("h4",{id:"cost"},"cost"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"@classmethod\ndef cost(cls, response: dict)\n")),(0,l.kt)("p",null,"Compute the cost of an API call."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"response")," ",(0,l.kt)("em",{parentName:"li"},"dict")," - The response from OpenAI API.")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("p",null,"  The cost in USD. 0 if the model is not supported."),(0,l.kt)("h4",{id:"extract_text"},"extract","_","text"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"@classmethod\ndef extract_text(cls, response: dict) -> List[str]\n")),(0,l.kt)("p",null,"Extract the text from a completion or chat response."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"response")," ",(0,l.kt)("em",{parentName:"li"},"dict")," - The response from OpenAI API.")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("p",null,"  A list of text in the responses."),(0,l.kt)("h4",{id:"extract_text_or_function_call"},"extract","_","text","_","or","_","function","_","call"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"@classmethod\ndef extract_text_or_function_call(cls, response: dict) -> List[str]\n")),(0,l.kt)("p",null,"Extract the text or function calls from a completion or chat response."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"response")," ",(0,l.kt)("em",{parentName:"li"},"dict")," - The response from OpenAI API.")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("p",null,"  A list of text or function calls in the responses."),(0,l.kt)("h4",{id:"logged_history"},"logged","_","history"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"@classmethod\n@property\ndef logged_history(cls) -> Dict\n")),(0,l.kt)("p",null,"Return the book keeping dictionary."),(0,l.kt)("h4",{id:"start_logging"},"start","_","logging"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"@classmethod\ndef start_logging(cls, history_dict: Optional[Dict] = None, compact: Optional[bool] = True, reset_counter: Optional[bool] = True)\n")),(0,l.kt)("p",null,"Start book keeping."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"history_dict")," ",(0,l.kt)("em",{parentName:"li"},"Dict")," - A dictionary for book keeping.\nIf no provided, a new one will be created."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"compact")," ",(0,l.kt)("em",{parentName:"li"},"bool")," - Whether to keep the history dictionary compact.\nCompact history contains one key per conversation, and the value is a dictionary\nlike:")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'{\n    "create_at": [0, 1],\n    "cost": [0.1, 0.2],\n}\n')),(0,l.kt)("p",null,'  where "created_at" is the index of API calls indicating the order of all the calls,\nand "cost" is the cost of each call. This example shows that the conversation is based\non two API calls. The compact format is useful for condensing the history of a conversation.\nIf compact is False, the history dictionary will contain all the API calls: the key\nis the index of the API call, and the value is a dictionary like:'),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'{\n    "request": request_dict,\n    "response": response_dict,\n}\n')),(0,l.kt)("p",null,"  where request_dict is the request sent to OpenAI API, and response_dict is the response.\nFor a conversation containing two API calls, the non-compact history dictionary will be like:"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'{\n    0: {\n        "request": request_dict_0,\n        "response": response_dict_0,\n    },\n    1: {\n        "request": request_dict_1,\n        "response": response_dict_1,\n    },\n')),(0,l.kt)("p",null,"  The first request's messages plus the response is equal to the second request's messages.\nFor a conversation with many turns, the non-compact history dictionary has a quadratic size\nwhile the compact history dict has a linear size."),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"reset_counter")," ",(0,l.kt)("em",{parentName:"li"},"bool")," - whether to reset the counter of the number of API calls.")),(0,l.kt)("h4",{id:"stop_logging"},"stop","_","logging"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"@classmethod\ndef stop_logging(cls)\n")),(0,l.kt)("p",null,"End book keeping."),(0,l.kt)("h2",{id:"chatcompletion-objects"},"ChatCompletion Objects"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"class ChatCompletion(Completion)\n")),(0,l.kt)("p",null,"A class for OpenAI API ChatCompletion."))}c.isMDXComponent=!0}}]);