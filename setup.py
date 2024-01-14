from setuptools import setup, find_packages
setup(
    name='visual-imitation-learning',
    version='1.0.0',
    author='Jianfeng Gao',
    author_email='jianfeng.gao@kit.edu',
    description="This packages contains modules for visual imitation learning",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'inquirer',
        'pathos',
        'paramiko',  # required for ssh connection, copy files, etc
        'scp',
        # 'pytorch3d'
    ],
    entry_points={
        "console_scripts": [
            "kvil_annotate=vil.perception.annotate_seg:main",
            "kvil=vil.kvil.run:main",
            "kvil_demo=vil.perception.demo_preprocessing:main",
            "kvil_viz_demo=vil.perception.viz.viz_demo:main",
            "kvil_sim=vil.deploy.bimanual_sim_armar6:main",
            "kvil_task=vil.deploy.kvil_task:main",
            "kvil_comp=vil.deploy.kvil_component:main",
            "kvil_trigger=vil.deploy.test_component:main",
            "kvil_hg=vil.deploy.create_handgroup_config:main",
        ]
    }
)
