## Evaluating Name Bias in Election Results: Machine Learning Project
### Marshall Carey-Matthews
#### Data Needed:
- "1976-2024-house.csv": contains data on US House Elections (1976-2024)
- "1976-2024-senate.csv": contains data on US Senate Elections (1976-2024)
- "Names2020_FirstNames_RaceHispanic.csv": Census Data on Name-Race Frequency
- "Names2020_FirstNames_Sex.csv": Census Data on Name-Sex Frequency
- "Names2020_LastNames_RaceHispanic.csv": Census Data on Name-Race Frequency


#### Note: This project was created in Google Colab (https://colab.research.google.com/drive/1JyRLtpelCSr3UZy1RZ1bF7UQPE_Pm29J?usp=sharing)
#### The proceeding README outlines my steps in creating and running the Ridge regression model exactly as they appear in the Colab notebook.

**Step 1: Reconciling Elections Data**

The MIT House and Senate data have slightly different categories, so first those categories need to be reconciled. Additionally, stripping and typecasting the data for standardization is also essential.

**Step 2: Pre-processing Census Names Data**

The Census names data is used to identify the White frequency and Male frequency of the candidates' names. Each name in the Census tables are given a neat frequency for each statistic. Then, the candidates' names in the House and Senate datasets have to be processed in such a way that they reveal a single first name and a last name for each candidate so that the candidates can be assigned a "minority value" for their names. This is one of the major feature engineering hurdles for this project.

**Step 3: Feature Engineer Democrat Vote Percentage Label**

The label being predicted by the machine learning model is intended to reflect how much candidates are hurt by voter bias relating to their name. However, trying to get a model to predict this based on simple voter percentage won't be effective because elections are zero sum, and to earn more votes, you have to take votes away from another candidate. Therefore, under a theoretical "bias-only" world, the name "John Smith" performs differently against "Mary Smith" as it does against "James Smith." The vectors have to be a race, not just a lone candidate.

**Step 4: Feature Engineer Incumbency**

To do this, we find the winner of the two-party table for each race. Then we look backward to the previous race and assess whether the winner of that race has the same name as one of the candidates. A Democrat-incumbent race gets +1 for incumbency, a Republican-incumbent race gets -1 for incumbency, and a race with no incumbents gets 0. This unfortunately gets way more complicated as Senate seats look back 6 years (not 2) and both districts are simply labeled "STATEWIDE", and some incumbents may run uncontested before later running contested.

**Step 5: Feature Engineer Partisanship Controls**

In this machine learning model, meaningfully controlling for the natural partisan lean of a particular district is very challenging. The urban-rural partisan divide is so strong that urban districts need to be controlled for, but statewide trends have to also be captured because states can shift over time. Therefore, I created a control feature of the Democratic vote share from the *previous* election.

**Step 6: Split Train & Test Sets**

Splitting the training and testing sets is challenging for this particular problem because well-known candidates will run for the same seat over many, many years (for example, Nancy Pelosi has been in office for 40 years). If those candidates (or anyone they ran against) were split between training and test sets, the test results would essentially be poisoned, since testing the same person you trained on is essentially a data leakage even if the election was technically different.

I used Gemini to explain how to group the elections touched by a single candidate such that the training/test split doesn't cause any poisioning. The solution is essentially to make a graph where a candidate is a node and an election matchup is an edge. When you split, each connected component should stay completely in one category. The result is not exactly 20% test data, but hopefully the solution is relatively effective.

**Step 7: Run the Model!**

Finally! For this project, I used the SciKit Ridge linear model. This model uses Ridge regression (i.e. L2 regularization) to achieve its output. Because I don't want to cut down on the amount of training data, I used 5-fold cross validation so that I could tune my regularization strength based on the average performance across each fifth of the training data.




**DIAGNOSTICS AND SUBGROUP ANALYSIS**

I was a little surprised at just how non-influential names were to the election outcomes, so I wanted to dig deeper. The first thing that surprised me was how high the "ideal" regularization rate was. I am curious if less regularization raises or suppresses name weight. If they suppress name weight, it only adds evidence that names do not have a high level of influence. Secondly, I am curious if the model was more predictive on particular subgroups, as perhaps name bias is more important in, for example, races without an incumbent for which voters might be weighing their decision on superficial name features more than on trust in the individual.

**OTHER MODEL INVESTIGATIONS**

To further elucidate the extent to which name bias plays a predictable role in influencing election outcomes, create other metrics to compare to my Ridge regression design results. This cell contains three methods: first, a "dummy regression," which basically just means it predicts every election outcome to be the training mean. Second, a more targeted heuristic that predicts the outcome based on the same district's previous result. Third, another Ridge regression that does not use any Census name frequencies. The first two methods reveal whether my original model was genuinely useful in predicting election outcomes, and the third reveals whether name bias (probably) had anything to do with the (general) election outcomes. Separately, since I deviated from my proposal by not creating a separate one-hot vector for state crossed with year, I wanted to see if adding that feature in would meaningfully change the predictiveness of the model.


#### AI USE DISCLOSURE

Gemini Flash 3.8 used for the following tasks relating to the code in GitHub:
- Generating an initial parser to glean first and last names from the elections data. I had to heavily modify the output created by Gemini as it included features that were antithetical to my design (such as converting nicknames into prototpyical first names) and did not include features that I needed for later processes in my design, such as a full name token creator for my incumbency function
- Suggesting solution to my issue of incumbency miscalculation for Senate races, which I implemented. I determined that its solution was relatively successful as the number of incumbent-having races found by my code greatly increased to a number closer to reality (in 2026, ~85% of seats had an incumbent, so a vast majority is the expected proportion for matchups with the incumbency flag raised).
- Creating the basic structure of networkx to create the graph of matchups so I could keep components on on one side of a split (since I have no experience with networkx), which I modified to interact with my k-fold. My validation printed under the Step 7 section verifies that the method created by Gemini to group candidates and the chain of races they influenced was able to prevent data leakage and still allow fair randomness and ~80/20 splitting
