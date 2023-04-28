 
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt 
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image



st.set_page_config(layout="wide")


@st.cache
def load_data(path):
    df = pd.read_csv(path)
    return df
df = load_data('Train.csv')

st.markdown("<h1 style = 'text-align: center; color: white ;'> E-Commerce shipping data </h1>", unsafe_allow_html = True)

tabs = st.tabs(['Home','Dataset preview', "Dataset Information", "Dataset findings", "visualizations", "Prediction using Gradient Boosting classifier model"])
Home_button = tabs[0]
Preview_button = tabs[1]
Info_button = tabs[2]
visual_button = tabs[4]
Find_button = tabs[3]
Predictions = tabs[5]


def attributeInformation(val):
    st.write('Below is the infomormation for the attribute  ' + val)
    st.write(df[val])
    temp = list(df[val].shape)
    st.markdown(f' The number of rows in the selected attribute  **{temp[0]}**')
    nullVal = df[val].isnull().sum()
    st.markdown(f' The number of null records present here is  **{nullVal}**')
    st.write('preview of null data from dataset:')
    st.write(df[val].isnull().sum())
    temp = type(df[val].iloc[2])
    if temp is np.int64:
        x = 'integer'
    elif temp is str:
        x = 'string'
    else:
        x = 'Undefined'
    st.write('The type of the data present is ' + x)

with Home_button:
    st.title("project Description")
    st.write("Welcome to our exciting e-commerce data analysis project! Our team has taken on the challenge of analyzing shipping data collected from an e-commerce company to gain insights into the company's shipping process and the effectiveness of delivering products on time."
    )
    st.write("We first extracted data related to shipping, such as the block, ship_method, num_calls, cost, gender, discount, weight, and on_time. We then used Python programming to perform data cleaning, manipulation, and statistical analysis to understand how well the company is delivering products on time."
    )
    st.write("Our team used various visualization tools, such as Altair, to create insightful and easy-to-read graphs, depicting how different variables affect shipping times. We looked at various factors that could affect on-time delivery, such as weight, shipping cost, and gender. By examining the data, we could identify trends, patterns, and anomalies in the shipping process, allowing the company to improve their shipping practices."
    )
    st.write("To make it easy for stakeholders to interact with our findings, we developed a user-friendly streamlit app that displays the results of our analysis.Our app also includes interactive visualizations that display the relationships between multiple attributes and one thing effecting other, allowing users to explore the trends and patterns in the data. They can interact with the graphs to zoom in and out, filter the data, and hover over the data points to view detailed information."
    )
    st.write("Moreover, our project presents a summary of our analysis findings, highlighting the key insights and trends that emerged from the data. Users can easily access this information and gain a deeper understanding of the factors that impact on-time delivery and the areas for improvement in the shipping process."
    )
    st.write("In conclusion, our e-commerce shipping analysis project provides the company with the insights necessary to improve their shipping processes and ultimately enhance customer satisfaction."
    )

    st.subheader("Team members:")
    st.subheader("Siddharth Jayachandra Babu     801252829")
    st.subheader("Deeksha Reddy Ganta      801311836")
    st.subheader("Sri Keshava Reddy     801328223")
    st.subheader("Mateo")


with Preview_button:
    st.title("Data information and filteration and Cleaning")
    st.write("We do not need to worry about redundant data since we do not have any. We have also verified this fact in the previous tabs")
    df.rename(columns={'Warehouse_block': 'block',
                   'Mode_of_Shipment': 'ship_method',
                   'Customer_care_calls': 'num_calls',
                   'Customer_rating': 'rating',
                   'Cost_of_the_Product': 'cost',
                   'Prior_purchases': 'num_prev_orders',
                   'Product_importance': 'priority',
                   'Gender': 'gender',
                   'Discount_offered': 'discount',
                   'Weight_in_gms': 'weight',
                   'Reached.on.Time_Y.N': 'on_time'},
          inplace=True)
    
    st.write(" The replacement of attribute names  has been done in accordance to the below description")
    st.write('Warehouse_block ----------block')
    st.write('Mode_of_Shipment ---------- ship_method')
    st.write('Customer_care_calls --------- num_calls')
    st.write('Customer_rating --------- rating')
    st.write('Prior_purchases -------num_prev_orders')
    st.write('Product_importance --------- priority')
    st.write('Gender -------- gender')
    st.write('Discount_offered -------- discount')
    st.write('Weight_in_gms ---------- weight')
    st.write('Cost_of_the_Product --------- cost')
    st.write('Reached.on.Time_Y.N -------- on_time')
    st.title("Dataset preview")
    st.write(" To get more readability to our available data set let us replace the dataframe's attribute names as per our convinience")
    st.write(df)
    st.write('Preview of null data from dataset')
    st.write("As we can see below we do not have any null data")
    st.write(df.isnull().sum())

with Info_button:
    

    array = list(df.shape)
    st.write('<style>body {font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 16px;}</style>', unsafe_allow_html=True)
    st.markdown(f' The number of rows in our dataset are **{array[0]}**')
    st.markdown(f'The number of coloumns in our dataset are **{array[1]}**')
    
   
    nav_option = st.selectbox('Choose your desired attributes to know about its information',
                                    ('block', 'ship_method', 'num_calls', 'rating', 'cost', 'num_prev_orders', 'priority', 'gender', 'discount', 'weight', 'on_time' ))

    # Define the content for each navigation option
    if nav_option == 'block':
        attributeInformation('block')
    elif nav_option == 'ship_method':
        attributeInformation('ship_method')
    elif nav_option == 'num_calls':
        attributeInformation('num_calls')
    elif nav_option == 'rating':
        attributeInformation('rating')
    elif nav_option == 'cost':
        attributeInformation('cost')
    elif nav_option == 'num_prev_orders':
        attributeInformation('num_prev_orders')
    elif nav_option == 'priority':
        attributeInformation('priority')
    elif nav_option == 'gender':
        attributeInformation('gender')
    elif nav_option == 'discount':
        attributeInformation('discount')
    elif nav_option == 'weight':
        attributeInformation('weight')
    elif nav_option == 'Reached.on.Time_Y.N':
        attributeInformation('Reached.on.Time_Y.N')



with visual_button:
    st.title("Dataset Visualizations")
    tabs = st.tabs(["Bar Chart", "Correlation heat map", "Density graphs with box plots", "Pie chart with parallel bar chart" ])
    with tabs[0]:
        grouped = df.groupby('cost').agg({'discount': 'mean'})
        grouped = grouped.reset_index()
        cost_filter = alt.binding_range(min=0, max=df['cost'].max(), step=1, name='Cost Filter:')
        filter_slider = alt.selection_single(bind=cost_filter, fields=['cost'])
        chart = alt.Chart(grouped).mark_bar().encode(
            x=alt.X('cost', title='Cost'),
            y=alt.Y('discount', title='Discount'),
            color=alt.Color('discount', legend=None)
        ).properties(
            title='Dependancies between price of product and the discount offered',
            width=800,
            height=600
        ).configure_scale(
            bandPaddingInner=0
        ).add_selection(
            filter_slider
        ).transform_filter(
            filter_slider
        ).interactive()
        chart
        st.write("Use this filter to know the specific details of a particular cost value")
        reset_button = st.button("Reset chart")
        if reset_button:
            pass
        
        st.subheader("From the above chart we find a trend where all the higher priced objects seem to have a low discount and the e-commerce platform from this can benifit from the fact that the dealers who sell through their e-commmerce site prefer to give more discount for lower priced objects")
        st.subheader("This can clearly be understood from the graph as we can see a Decline in the graph's discount as the product cost value grows")


    with tabs[1]:
        plt.figure(figsize = (8, 4))
        heat_map = sns.heatmap(df.corr(), annot = True, fmt = '0.2f', annot_kws = {'size' : 13}, linewidth = 5, linecolor = 'silver')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.subheader("Findings from heatmap")
        st.subheader("The correlation analysis shows that the discount offered is highly positively correlated with reaching the delivery on time, with a correlation coefficient of 40%. On the other hand, the weight of the product has a negative correlation of -27% with reaching the delivery on time. Furthermore, there is a negative correlation of -38% between the discount offered and the weight of the product, as well as a negative correlation of -28% between the number of customer care calls and the weight of the product. However, there is a positive correlation of 32% between the number of customer care calls and the cost of the product. Lastly, there is a slightly positive correlation between prior purchases and customer care calls.")

    with tabs[2]:
        st.subheader("Below we have two findings based on the discount and weight of the graph")
        st.text("note: We focus on the attribute Reached on Time because that is our target attribute")
        st.markdown("""---""")
        st.subheader("Part 1 - Inference from Density")
        st.write(" Discount Density graph")
        chart = alt.Chart(df).transform_density(
            'discount',
            as_=['discount', 'density'],
        ).mark_area(opacity=0.5, interpolate='step').encode(
            alt.X('discount:Q', bin=alt.Bin(maxbins=20), title='discount'),
            alt.Y('density:Q', stack=None, title='Density'),
           
        ).properties(width=800, height=500)
        chart
        st.subheader("We learn from the above density chart that low discounts have had a high density of occurence amongst the packages, with this detail let us find out if there could be a relation with the packages delivered on time")

        chart_box_disc = alt.Chart(df).mark_boxplot(color = 'white').encode(
            x='on_time:N',
            y='discount:Q',
            color='on_time:N'
        ).properties(
            width=800,
            height=500,
            title='Box Plot of Values based on package discount'
        )
        chart_box_disc
        st.write("0 -- packages not delivered on time                1 -- packages delivered on time")
        st.subheader("We can see from the above box plot that packages delivered on time have a varied discount value which spans from low to high, whereas the packages not delivered on time are restricted to low discounts and thus we can say that packages with discounts have been of priority with respect to the delivery time")
        st.markdown("""---""")

        st.subheader("Part 2 - Inference from weight")
        st.write("Density graph for weight in grams")
        chart_weight = alt.Chart(df).transform_density(
            'weight',
            as_=['weight', 'density'],
        ).mark_area(opacity=0.5, interpolate='step').encode(
            alt.X('weight:Q', bin=alt.Bin(maxbins=20), title='Weight in Grams'),
            alt.Y('density:Q', stack=None, title='Density'),
        ).properties(width=800, height=500)
        chart_weight
        st.subheader("We can see from the density graph that packages of high weight and packages of low weight have a high density, with this detail let us find out if there could be a relation with the packages delivered on time")

        chart_box_wt = alt.Chart(df).mark_boxplot(color = 'white').encode(
            x='on_time:N',
            y='weight:Q',
            color='on_time:N'
        ).properties(
            width=800,
            height=500,
            title='Box Plot of Values by the packages which reached on time'
        )
        chart_box_wt
        st.subheader("From the box plot we can see that packages of high weight and low weight have not been delivered on time and all the packages weighing in the mid range have been delivered on time. Thus we can conclude that packages with high and low weight have had issues in delivery")



    with tabs[3]:
        st.subheader("Below we use pie charts and parallel bar chart to infer various trends in our dataset. There are two parts one which focuses on the various blocks and the other with mode of shipment")
        st.text("note: We focus on the attribute Reached on Time because that is our target attribute")
        st.markdown("""---""")
        st.subheader("Part 1 - Using warehouse Blocks information")
        st.write("Pie chart to display the block with largest storage")
        pie_block_df = df.groupby('block').size().reset_index(name = 'count')
       
        pie_chart = alt.Chart(pie_block_df).mark_arc().encode(
            theta="count",
            color="block"
        )
        pie_chart

        st.subheader("From the above pie chart we can see that warehouse block F has the highest packages stores with the rest of the warehouses being evenly dirstributed with packages. Using this information let us find out if there could be any relation with the delivery time")   
        pivoted_df = df.groupby(['on_time','block']).size().reset_index(name = 'count')

        side_chart = alt.Chart(pivoted_df).mark_bar().encode(
            alt.Column('block',  header=alt.Header(labelColor='white')), alt.X('on_time'),
            alt.Y('count', axis=alt.Axis(grid=False)), 
            alt.Color('on_time')).properties(width = 100, height = 600)
        
        side_chart

        st.subheader("From the above side by side bar chart we can infer that warehouse F has had the highest number of packages delivered on time. The other warehouses have almost equal number of packages delivered and not delivered on time. But its pretty amazing to know that warehouse F has the highest packages stored and also delivered them with high success rate. This shows the efficiency of warehosue F")
        st.markdown("""---""")
        st.subheader("Part 2 - Using shipment details")
       
        st.write("Pie chart to display the mode of shipment with largest storage")
        ship_pie_block_df = df.groupby('ship_method').size().reset_index(name = 'count')
    
        ship_pie_chart = alt.Chart(ship_pie_block_df).mark_arc().encode(
            theta="count",
            color="ship_method"
        )
        ship_pie_chart
        st.subheader("From the above pie chart we can see that highest packages have been transported via ships. Using this information let us find out if there could be any relation with the delivery time")
    
        ship_pivoted_df = df.groupby(['on_time','ship_method']).size().reset_index(name = 'count')

        ship_side_chart = alt.Chart(ship_pivoted_df).mark_bar().encode(
            alt.Column('ship_method',header=alt.Header(labelColor='white')), alt.X('on_time'),
            alt.Y('count', axis=alt.Axis(grid=False)), 
            alt.Color('on_time')).properties(width = 100, height = 600)
        ship_side_chart
        st.subheader("From the above side by side bar chart we can see that the highest number of packages were delivered through shipment on time. Regardless of the large number of packages being transported through ships. It is proved that ships have served efficient in delivering packages on time")

       
                

with Find_button:


    st.title("Dataset Findings")
    warehouse_group = df.groupby('block').size().reset_index(name='count')
    chart = alt.Chart(warehouse_group).mark_bar().encode(
        x=alt.X('block:N', title='block'),
        y=alt.Y('count:Q', title='Count'),
        tooltip=['block', 'count']
    ).properties(
        title='Block Counts',
        width = 500,
        height = 400
    )
   
    shipment_group = df.groupby('ship_method').size().reset_index(name = 'count')
    ship_chart = alt.Chart(shipment_group).mark_bar().encode(
        x=alt.X('ship_method:N', title='ship_method'),
        y=alt.Y('count:Q', title='Count'),
        tooltip=['ship_method', 'count']
    ).properties(
        title='shipment counts',
        width = 500,
        height = 400
    )
   

    customer_care_calls = df.groupby('num_calls').size().reset_index(name = 'count')
    calls_chart = alt.Chart(customer_care_calls).mark_bar().encode(
        x=alt.X('num_calls:N', title='num_calls'),
        y=alt.Y('count:Q', title='Count'),
        tooltip=['num_calls', 'count']
    ).properties(
        title='No of customer care calls - total count',
        width = 500,
        height = 400
    )
   

    rating_count = df.groupby('rating').size().reset_index(name = 'count')
    rating_chart = alt.Chart(rating_count).mark_bar().encode(
        x=alt.X('rating:N', title='rating'),
        y=alt.Y('count:Q', title='Count'),
        tooltip=['rating', 'count']
    ).properties(
        title='Total rating',
        width = 500,
        height = 400
    )
   

    prior_purchase_count = df.groupby('num_prev_orders').size().reset_index(name = 'count')
    purchase_chart = alt.Chart(prior_purchase_count).mark_bar().encode(
        x=alt.X('num_prev_orders:N', title='num_prev_orders'),
        y=alt.Y('count:Q', title='Count'),
        tooltip=['num_prev_orders', 'count']
    ).properties(
        title='Number of previous orders count',
        width = 500,
        height = 400
    )
    

    Reached_time_count = df.groupby('on_time').size().reset_index(name = 'count')
    on_time_chart = alt.Chart(Reached_time_count).mark_bar().encode(
        x=alt.X('on_time:N', title='on_time'),
        y=alt.Y('count:Q', title='Count'),
        tooltip=['on_time', 'count']
    ).properties(
        title='Reached on time count',
        width = 500,
        height = 400
    )
   


    Gender_count = df.groupby('gender').size().reset_index(name = 'count')
    gender_chart = alt.Chart(Gender_count).mark_bar().encode(
        x=alt.X('gender:N', title='gender'),
        y=alt.Y('count:Q', title='Count'),
        tooltip=['gender', 'count']
    ).properties(
        title='Gender count',
        width = 500,
        height = 400
    )
   

    priority_count = df.groupby('priority').size().reset_index(name = 'count')
    priority_chart = alt.Chart(priority_count).mark_bar().encode(
        x=alt.X('priority:N', title='num_prev_orders'),
        y=alt.Y('count:Q', title='Count'),
        tooltip=['priority', 'count']
    ).properties(
        title='Product importance count',
        width = 500,
        height = 400
    )
    

    row1 = alt.hconcat(chart, ship_chart, spacing = 200)
    row2 = alt.hconcat(rating_chart, calls_chart, spacing = 200)
    row3 = alt.hconcat(purchase_chart, on_time_chart, spacing = 200)
    row4 = alt.hconcat(gender_chart, priority_chart, spacing = 200)
    grid = alt.vconcat(row1,row2, row3, row4)
    grid
    st.subheader("We have multiple findings that we can infer from the above charts")
    st.write("* From block counts chart we can know that block F has the highest counts of packages stored")
    st.write("* From shipment counts chart we can infer that the highest packages are been delivered via ships")
    st.write("* From the total rating chart we can see that all products have equal dirstribution in rating from 1 to 5 with over 2000 ratings per star. Yet the highest rating is with 3 star ratings")
    st.write("* From the number of customer care calls chart we can see that an average of 4 calls for a product has the highest count")
    st.write("* From the number of previous order we can see that there has been previous orders for products and for a product to have three previous orders stands the highest")
    st.write("* From the reached on time count chart we can see that the packages have been delivered  on time majorly with a total count of 6563 packages, and 4436 packages have not been delivered on time")
    st.write("* From the gender chart we can see that female have higher orders over men with a count of 5545, and men have a count of 5454. There is not a significant difference between both.")
    st.write("* From product importance chart we can learn that producs with low importance are high in number and products with medium importance fall next with, high important products being less in number.")
    st.subheader("To know further details or learn more information feel free to hover over the charts and know the specific detail counts")

with Predictions:
    st.subheader("In our project we have used Gradient boosting classifier model to predict our target attribute Reached.on.Time_Y.N.")
    st.subheader("The reason for us to choose this model is because we believed for this large number of records and complex patterns involved it would be wise to use this model.")
    st.subheader("The source code and  predictions can be found in the jupyter notebook file submitted on canvas")
    st.subheader(" We were able to achieve a Train data accuracy of 0.7279890892559479 and a test data accuracy of 0.6915909090909091")
    