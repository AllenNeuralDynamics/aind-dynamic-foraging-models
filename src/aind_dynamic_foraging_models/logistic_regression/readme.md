## Choosing logistic regression models

### Su 2022

![Untitled](https://hanhou.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fe2dab5b7-862e-46a8-aa74-8194ed4315fc%2F54b9718d-6916-48ea-a337-550410a88254%2FUntitled.png?table=block&id=a2db5af7-f2d7-4485-af6c-01a0908546f6&spaceId=e2dab5b7-862e-46a8-aa74-8194ed4315fc&width=1340&userId=&cache=v2)

$$
logit(p(c_r)) \sim RewardedChoice+UnrewardedChoice
$$

### Bari 2019

![Untitled](https://hanhou.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fe2dab5b7-862e-46a8-aa74-8194ed4315fc%2F9965a743-89e5-4335-af09-927d96f304e3%2FUntitled.png?table=block&id=1010abe7-4a81-429d-b1b0-5730630e508e&spaceId=e2dab5b7-862e-46a8-aa74-8194ed4315fc&width=1150&userId=&cache=v2)

![Untitled](https://hanhou.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fe2dab5b7-862e-46a8-aa74-8194ed4315fc%2Fcb2bbc09-8032-4eb5-8a55-bdadf9f42078%2FUntitled.png?table=block&id=c5cf0499-df10-4ebe-9e81-7eb5e504eede&spaceId=e2dab5b7-862e-46a8-aa74-8194ed4315fc&width=1150&userId=&cache=v2)

$$
logit(p(c_r)) \sim RewardedChoice+Choice
$$

### Hattori 2019

![Untitled](https://hanhou.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fe2dab5b7-862e-46a8-aa74-8194ed4315fc%2F44b49866-9f22-45fa-95db-0287a5a9bcfe%2FUntitled.png?table=block&id=20531979-9296-4b51-a41b-bab2e8615c84&spaceId=e2dab5b7-862e-46a8-aa74-8194ed4315fc&width=1340&userId=&cache=v2)

$$
logit(p(c_r)) \sim RewardedChoice+UnrewardedChoice+Choice
$$

### Miller 2021

![Untitled](https://hanhou.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fe2dab5b7-862e-46a8-aa74-8194ed4315fc%2F7cd393c3-8c0a-4b63-a4c6-e84a00dff21a%2FUntitled.png?table=block&id=31e3450e-d60f-4c2a-9883-da91a5eaed9b&spaceId=e2dab5b7-862e-46a8-aa74-8194ed4315fc&width=1250&userId=&cache=v2)

$$
logit(p(c_r)) \sim Choice + Reward+ Choice*Reward
$$

### Encodings

- Ignored trials are removed

| choice | reward | Choice | Reward | RewardedChoice | UnrewardedChoice | Choice * Reward |
| --- | --- | --- | --- | --- | --- | --- |
| L | yes | -1 | 1 | -1 | 0 | -1 |
| L | no | -1 | -1 | 0 | -1 | 1 |
| R | yes | 1 | 1 | 1 | 0 | 1 |
| L | yes | -1 | 1 | -1 | 0 | -1 |
| R | no | 1 | -1 | 0 | 1 | -1 |
| R | yes | 1 | 1 | 1 | 0 | 1 |
| L | no | -1 | -1 | 0 | -1 | 1 |

Some observations:

1. $RewardedChoice$ and $UnrewardedChoice$ are orthogonal
2. $Choice = RewardedChoice + UnrewardedChoice$
3. $Choice  * Reward = RewardedChoice -  UnrewardedChoice$

### Comparison

|  | Su 2022 | Bari 2019 | Hattori 2019 | Miller 2021 |
| --- | --- | --- | --- | --- |
| Equivalent to | RewC + UnrC | RewC + (RewC + UnrC) | RewC + UnrC + (RewC + UnrC) | (RewC + UnrC) + (RewC - UnrC) + Rew |
| Severity of multicollinearity | Not at all | Medium | Severe | Slight |
| Interpretation | Like a RL model with different learning rates on reward and unrewarded trials. | Like a RL model that only updates on rewarded trials, plus a choice kernel (tendency to repeat previous choices). | Like a RL model that has different learning rates on reward and unrewarded trials, plus a choice kernel (the full RL model from the same paper). | Like a RL model that has symmetric learning rates for rewarded and unrewarded trials, plus a choice kernel. However, the $Reward $ term seems to be a strawman assumption, as it means “if I get reward on any side, I’ll choose the right side more”, which doesn’t make much sense. |
| Conclusion | Probably the best | Okay | Not good due to the severe multicollinearity | Good |


## Regularization and optimization
The choice of optimizer depends on the penality term, as listed [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression).

- `lbfgs` - [`l2`, None]
- `liblinear` - [`l1`, `l2`]
- `newton-cg` - [`l2`, None]
- `newton-cholesky` - [`l2`, None]
- `sag` - [`l2`, None]
- `saga` - [`elasticnet`, `l1`, `l2`, None]