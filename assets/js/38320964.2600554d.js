"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[8308],{3905:(e,t,n)=>{n.d(t,{Zo:()=>u,kt:()=>f});var r=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function c(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var g=r.createContext({}),l=function(e){var t=r.useContext(g),n=t;return e&&(n="function"==typeof e?e(t):c(c({},t),e)),n},u=function(e){var t=l(e.components);return r.createElement(g.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},s=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,g=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),s=l(n),f=o,d=s["".concat(g,".").concat(f)]||s[f]||p[f]||a;return n?r.createElement(d,c(c({ref:t},u),{},{components:n})):r.createElement(d,c({ref:t},u))}));function f(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,c=new Array(a);c[0]=s;var i={};for(var g in t)hasOwnProperty.call(t,g)&&(i[g]=t[g]);i.originalType=e,i.mdxType="string"==typeof e?e:o,c[1]=i;for(var l=2;l<a;l++)c[l]=n[l];return r.createElement.apply(null,c)}return r.createElement.apply(null,n)}s.displayName="MDXCreateElement"},8806:(e,t,n)=>{n.r(t),n.d(t,{contentTitle:()=>c,default:()=>u,frontMatter:()=>a,metadata:()=>i,toc:()=>g});var r=n(7462),o=(n(7294),n(3905));const a={sidebar_label:"coding_agent",title:"autogen.agent.coding_agent"},c=void 0,i={unversionedId:"reference/autogen/agent/coding_agent",id:"reference/autogen/agent/coding_agent",isDocsHomePage:!1,title:"autogen.agent.coding_agent",description:"PythonAgent Objects",source:"@site/docs/reference/autogen/agent/coding_agent.md",sourceDirName:"reference/autogen/agent",slug:"/reference/autogen/agent/coding_agent",permalink:"/FLAML/docs/reference/autogen/agent/coding_agent",editUrl:"https://github.com/microsoft/FLAML/edit/main/website/docs/reference/autogen/agent/coding_agent.md",tags:[],version:"current",frontMatter:{sidebar_label:"coding_agent",title:"autogen.agent.coding_agent"},sidebar:"referenceSideBar",previous:{title:"agent",permalink:"/FLAML/docs/reference/autogen/agent/agent"},next:{title:"execution_agent",permalink:"/FLAML/docs/reference/autogen/agent/execution_agent"}},g=[{value:"PythonAgent Objects",id:"pythonagent-objects",children:[],level:2}],l={toc:g};function u(e){let{components:t,...n}=e;return(0,o.kt)("wrapper",(0,r.Z)({},l,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h2",{id:"pythonagent-objects"},"PythonAgent Objects"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"class PythonAgent(Agent)\n")),(0,o.kt)("p",null,"(Experimental) Suggest code blocks."))}u.isMDXComponent=!0}}]);