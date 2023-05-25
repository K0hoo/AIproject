# Menu Recommendation Project

## Overview
This project aims to recommend menus based on the ingredients available in the user's refrigerator. Instead of simply listing the ingredients, the focus is on providing high-quality recommendations that help users utilize their ingredients effectively.

The recommendation system is built upon the refinement of approximately 1,500 recipes obtained from a website called '만개의 레시피'. Each recipe is structured as follows:

1. Name
2. Number of views
3. Servings
4. Cooking time
5. Difficulty
6. URLs
7. Ingredients
8. Importance of each ingredient
9. Recipe

By considering the user's favorite foods, the system recommends similar dishes that can be prepared using the available ingredients. The similarity between dishes is determined based on the following criteria:

1. Recipe Similarity
2. Similarity of ingredients
3. Similarity of the dishes themselves

The system sorts the data based on similarity and selects the top 100 menus that can be prepared using the user's current ingredients. These menus are then presented to the user for their consideration.

## How to Use
To use this project, follow the steps below:

1. Provide a list of ingredients available in your refrigerator. This can be done manually or by integrating with an existing app or system that already tracks refrigerator contents.

2. Optionally, indicate your favorite foods to receive recommendations based on your preferences.

3. The system will process the ingredient data and calculate the similarities between dishes and ingredients.

4. Based on the calculated similarities, the system will present a list of 100 menus that can be prepared using the provided ingredients.

5. Review the recommended menus and select the ones that appeal to you.

6. Access the URLs provided for each menu to view the full recipe and cooking instructions.

7. Prepare and enjoy your chosen dish!

## Technical Details
The project utilizes a dataset of approximately 1,500 recipes sourced from 'Ten Thousand Recipes'. The dataset includes various attributes for each recipe, such as name, views, servings, cooking time, difficulty, URLs, ingredients, importance of each ingredient, and the recipe itself.

The recommendation system employs a combination of recipe similarity, ingredient similarity, and dish similarity to determine the most relevant menus for the user. The similarity calculations are performed using advanced algorithms tailored to this specific project.

## Future Enhancements
The project has the potential for further enhancements and improvements. Some ideas for future development include:

1. Integration with popular grocery shopping apps to automatically generate ingredient lists based on the user's purchases.

2. Incorporating user feedback and ratings to continuously improve the recommendation algorithm.

3. Allowing users to customize their dietary preferences (e.g., vegetarian, gluten-free) to receive tailored menu recommendations.

4. Providing nutritional information for each recommended dish to help users make informed choices.

5. Collaborating with local grocery stores or online food delivery services to enable seamless ingredient acquisition.

## Contributions and Feedback
Contributions and feedback are welcome to improve this project. If you have any ideas, bug reports, or suggestions for enhancements, please feel free to contribute to the project repository or reach out to the project team.

Thank you for using this menu recommendation project, and we hope it helps you make delicious and creative meals using the ingredients in your refrigerator!


## License
This project is licensed under the MIT License.