
# Chart

_temp_df_chart = new_df[['Simulation Type', '# Samples Evaluated/Interaction Number', 'Accuracy']]

new_df.to_pickle('new_df.pkl')
_temp_df_chart.to_pickle('_temp_df_chart.pkl')


sns.set(rc={'figure.figsize':(15.7,8.27)})
# palette=dict(setosa="#9b59b6", virginica="#3498db", versicolor="#95a5a6")
palette=_list_simulation_sample_pallete


sns.lineplot(data=_temp_df_chart, 
             x="# Samples Evaluated/Interaction Number", 
             y="Accuracy", 
             hue="Simulation Type",
             palette=palette
            )
savefig            
plt.savefig('save_as_a_png.png')