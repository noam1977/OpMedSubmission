
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv("surgeries to predict.csv")

doctorHist = df.DoctorID.value_counts().value_counts()
anesHist = df.AnaesthetistID.value_counts().value_counts()
print(f"DoctorID - Number of reoccurances:\n{doctorHist}")
print(f"AnaesthetistID - Number of reoccurances:\n{anesHist}")
y = df['Duration in Minutes']
groups = []
groups.append(df.groupby(['Surgery Type','Anesthesia Type']))
groups.append(df.groupby(['Surgery Type']))
groups.append(df.groupby(['Anesthesia Type']))

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
fig = plt.figure()
markers = ['o', 'v', 's', 'p', '*', 'h', 'H', '+', 'x', 'D']
titles = ['Surgery Type and Anesthesia Type', 'Surgery Type', 'Anesthesia Type']

for sb,group in zip([231,232,233],groups):
    ax = fig.add_subplot(sb, projection='3d')

    for i,g in enumerate(group):
        x,y,z = g[1]['Age'], g[1]['BMI'],g[1]['Duration in Minutes']
        ax.scatter(x, y, z, c=colors[i], marker=markers[i])

    ax.set_xlabel('Age')
    ax.set_ylabel('BMI')
    ax.set_zlabel('Duration in Minutes')
    ax.title.set_text(titles.pop(0))

ax = fig.add_subplot(257, projection='3d')
# plot only group 3
group = groups[0]
g = list(group)[6]
x,y,z = g[1]['Age'], g[1]['BMI'],g[1]['Duration in Minutes']
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Duration in Minutes')

ax = fig.add_subplot(259, projection='3d')
# plot only group 3
group = groups[0]
g = list(group)[6]
x,y,z = g[1]['Age'], g[1]['BMI'],g[1]['Duration in Minutes']
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Duration in Minutes')

plt.show()






#
# allx = []
# ally = []
# allz = []
# for g in groups:
#     z = g[1]['Duration in Minutes']
#     z = z-z.mean()
#     allz.extend(z)
#     x = g[1]['Age']
#     y = g[1]['BMI']
#     allx.extend(x)
#     ally.extend(y)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('Age')
# ax.set_ylabel('BMI')
# ax.set_zlabel('Duration in Minutes')
# plt.show()
# for g in groups:
#     # plot 3d graph for each group, x = age, y = bmi, z = duration in minutes
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(g[1]['Age'], g[1]['BMI'], g[1]['Duration in Minutes'])
#     ax.set_xlabel('Age')
#     ax.set_ylabel('BMI')
#     ax.set_zlabel('Duration in Minutes')
#     plt.show()
#
#
#
#
