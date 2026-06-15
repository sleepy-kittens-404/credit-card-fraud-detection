from setuptools import find_packages,setup
HYPHEN_E_DOT = '-e .'
def get_requirements(file_path):
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
setup(
    name= 'credit_card_fraud_detection',
    author="Muhammad Sami",
    version='0.0.1',
    author_email='muhamadsami501@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')


)