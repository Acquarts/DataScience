
# Airlines Flights Dataset for Different Cities

The **Flights Booking Dataset** of various airlines is scraped date-wise from a well-known website in a structured format. The dataset contains records of flight travel details between cities in India, including features such as **Source & Destination City**, **Arrival & Departure Time**, **Duration**, and **Price**.

This data is available as a **CSV file** and is analyzed using the **Pandas DataFrame**.

This analysis is helpful for professionals working in the **Airlines** and **Travel** domains.

Using this dataset, we answered multiple questions with Python in our project.

## Questions Addressed

- **Q.1.** What are the airlines in the dataset, accompanied by their frequencies?
- **Q.2.** Show bar graphs representing the Departure Time & Arrival Time.
- **Q.3.** Show bar graphs representing the Source City & Destination City.
- **Q.4.** Does price vary with airlines?
- **Q.5.** Does ticket price change based on the departure time and arrival time?
- **Q.6.** How does the price change with different Source and Destination cities?
- **Q.7.** How is the price affected when tickets are bought just 1 or 2 days before departure?
- **Q.8.** How does the ticket price vary between Economy and Business class?
- **Q.9.** What will be the average price of Vistara airline for a flight from Delhi to Hyderabad in Business Class?

## Main Features / Columns

1. **Airline**  
   The name of the airline company.  
   _Type:_ Categorical (6 different airlines)

2. **Flight**  
   The plane's flight code.  
   _Type:_ Categorical

3. **Source City**  
   City from which the flight takes off.  
   _Type:_ Categorical (6 unique cities)

4. **Departure Time**  
   Derived categorical feature created by binning time periods.  
   _Type:_ Categorical (6 time labels)

5. **Stops**  
   Number of stops between source and destination cities.  
   _Type:_ Categorical (3 distinct values)

6. **Arrival Time**  
   Derived categorical feature created by binning time intervals.  
   _Type:_ Categorical (6 time labels)

7. **Destination City**  
   City where the flight will land.  
   _Type:_ Categorical (6 unique cities)

8. **Class**  
   Seat class of the ticket.  
   _Type:_ Categorical (Business, Economy)

9. **Duration**  
   Total travel time between cities in hours.  
   _Type:_ Continuous

10. **Days Left**  
    Days between the booking date and the trip date.  
    _Type:_ Derived numerical feature

11. **Price**  
    Ticket price.  
    _Type:_ Target variable
