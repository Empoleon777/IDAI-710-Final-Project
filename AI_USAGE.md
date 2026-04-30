# Billy
My ChatGPT usage revolved around what it usually does for the vast majority of coding projects - Debugging. Specifically, when writing my LSTM code, I started from an online tutorial on a similar problem to get an idea of what the basic syntax might look like, as I've never built an algorithm like this myself before.

Then, during the testing phase, If I was concerned about whether I was going in the right direction, I would show ChatGPT my code, along with a detailed explanation of the problem (To the best of my ability, at least; explanations are not my strongest suit), and if I encountered an error, I would show it the log of said error, then modify my code accordingly to fix the problem. Whenever possible, I would attempt to solve problems myself, but ChatGPT definitely saved me potential hours banging my head against the keyboard to find where the breakage is occurring, and how to fix it.

I learned a fair amount about how to build and properly handle an LSTM model from all of this, which was honestly a good experience. I might be able to make use of this later. Another big thing I learned was how to undo feature scaling, a crucial skill for two of the models we were working with; XGBoost doesn't need it, but it's somewhat important for the Linear Regression and especially LSTM. Speaking of which, another crucial thing I learned is that you need to scale your training and test sets separately, or leakage occurs.

As I said above, the main thing ChatGPT helped me with was debugging, knowing proper syntax to achieve certain tasks, and occasionally, guidance on the proper next steps for specific parts of the process (Such as when I was concerned about my feature engineering; Enzo and Valbona both worked with more features than I did, while I simply used the ones included in the set by default. I decided to ask ChatGPT what it would recommend I do, showing it the exact code both of them used for their feature engineering).

Relevant chats: 
* https://chatgpt.com/share/69f051df-f8f4-83ea-8737-aa5e83509699
* https://chatgpt.com/share/69f2c5ff-8fac-83ea-b86c-cf4e672fc393
* https://chatgpt.com/share/69f2c60a-4cd4-83ea-8b27-16472eec234d
* https://chatgpt.com/share/69f2c5e4-42b4-83ea-841f-cf1c875472b9
* https://chatgpt.com/share/69f2c615-0d1c-83ea-b2b2-6dcc611b1d9b

# Valbona

# Enzo