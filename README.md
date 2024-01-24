[![Contributors][contributors-shield]][contributors-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://chat.openai.com/">
    <img src="https://media.sketchfab.com/models/901f21ea7d0b46748e8026b6b5f7306d/thumbnails/094af3f46ce14da0b4045a50545d1d15/3cb664da18494ecb9e9e2408d0fdd2a3.jpeg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">END TO END GPT FROM SCRATCH</h3>

  <p align="center">
    This repository is designed for educational purposes, providing a comprehensive guide on constructing a GPT model from scratch using PyTorch, training the model with textual data, and deploying it.
    <br />
    <a href="https://docs.google.com/document/d/1HgSY3pI5c_SjbEVn77Ve8zJ8VWmXwi9RkZY4czJx-Qg/edit"><strong>Explore the docs »</strong></a>
    <br />
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

This project is for educational purpose.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Next][python]][python-url]
* [![React][Flask]][sanic-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started

This project has developed for platform independent using contairization.

### Prerequisites

Ensure that Python 3.9 is installed on your system. If you are using a Mac, refer to the provided blog link for detailed
instructions on installing a specific Python version. Please review the information in the blog for
guidance. [python installation](https://www.freecodecamp.org/news/python-version-on-mac-update/)

### Installation  (Mac)

1. Clone the repo  
`git clone https://github.com/navanith007/gpt_from_scratch.git`

2 Create python virtual environement

`python3.9 -m venv llm_Env`

`source llm_Env/bin/activate`

3. Install the requirements

`pip install -r requirements.txt`

4. Running the gpt service locally

`sh start_app_service.sh`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

To make changes to this service you need follow below steps before going into production

1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
2. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
3. merge your changes to preprod and get it QC.
3. Push to the Branch (`git push origin feature/AmazingFeature`)
4. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/navanith-sci-dev/repo_name.svg?style=for-the-badge

[contributors-url]: https://github.com/bombinatetech/ml-content-services/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge

[forks-url]: https://github.com/github_username/repo_name/network/members

[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge

[stars-url]: https://github.com/github_username/repo_name/stargazers

[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge

[issues-url]: https://github.com/github_username/repo_name/issues

[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge

[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/linkedin_username

[product-screenshot]: images/screenshot.png

[python]: https://img.shields.io/badge/python-000000?style=for-the-badge&logo=python&logoColor=python

[python-url]: https://www.python.org/

[FLask]: https://img.shields.io/badge/Flask-green?style=for-the-badge&logo=flask&logoColor=black

[sanic-url]: https://sanic.dev/en/
