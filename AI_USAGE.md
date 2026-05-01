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
My Claude usage was mainly around debugging the optimization component. When my code was not producing the results I expected, I would share the error or the output with Claude to get a better understanding of what was going wrong. Claude would suggest fixes and explain why the issue was occurring, but I would then adapt those suggestions to fit our specific case rather than copying them directly. For example when the optimization results were not aligning with what the paper described, I used Claude to think through the logic and adjust the implementation accordingly. 

I also made use of ChatGPT throughout the project, mainly for understanding explanations and concepts that were difficult to grasp, since English is my second language. Whenever I encountered a passage or a technical explanation that was hard to follow, ChatGPT helped me understand it in a clearer and more accessible way, which made it significantly easier to contribute meaningfully to both the code and the paper.


# Enzo
I also mainly used AI to debug very unexpected errors in Python. Not knowing in detail some libraries so it was very useful to me, especially for data processing with libraries like pandas and sklearn.

I also used it sometimes for translating text coming from the web or some parts that I had first written in my original language. Concerning some passages that are very difficult to explain in another language without losing details.

It also helped me understand why I was getting negative demands even though they could not mathematically be negative. To understand that it was actually possible and that it was just a model trying to represent reality in the end.
By adding a lot of new features, the model was never more efficient and even worse. I felt lost, but it was the addition of redundant and unnecessary data that was causing all of this.
