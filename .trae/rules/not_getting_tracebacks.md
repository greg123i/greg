---
title: not getting tracebacks
alwaysapply: true
---

 - before you generate any piece of code, think what it should do, if you think it will cause a problem, try ssomething else

 - no manager-itis and god objects: i dont want to see a NounActivityer() class that does everything, everywhere, all at once, keep the class or function names descriptive, else, you will turn the classes and/or functions into a mess that noone can understand

 - no classes, functions, methods, etc should do more than one thing, if thats the case, split them up

 - dont repeat yourself: if you see code that is repeated, make it a function or class and add it to the 'lib' folder at the root of the project or in a folder within the 'lib' folder if it is a specific type of code

 - organize stuff via folders: if you dont, it turns into an unorganized mess

 - refactor things: if its too long or too slow, refactor it