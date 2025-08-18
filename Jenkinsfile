pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/JedhaBootcamp/sample-ml-workflow.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    // Build Docker image using Jenkins Docker pipeline syntax
                    docker.build('ml-pipeline-image')
                }
            }
        }

        stage('Run Tests Inside Docker Container') {
            steps {
                withCredentials([
                    string(credentialsId: 'mlflow-tracking-uri', variable: 'MLFLOW_TRACKING_URI'),
                    string(credentialsId: 'aws-access-key', variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'aws-secret-key', variable: 'AWS_SECRET_ACCESS_KEY'),
                    string(credentialsId: 'backend-store-uri', variable: 'BACKEND_STORE_URI'),
                    string(credentialsId: 'artifact-root', variable: 'ARTIFACT_ROOT')
                ]) {
                    script {
                        // Run tests inside Docker using the Jenkins docker.image().inside block
                        docker.image('ml-pipeline-image').inside('-e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI ' +
                                                               '-e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID ' +
                                                               '-e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY ' +
                                                               '-e BACKEND_STORE_URI=$BACKEND_STORE_URI ' +
                                                               '-e ARTIFACT_ROOT=$ARTIFACT_ROOT') {
                            // Execute pytest without using sh
                            bat 'pytest --maxfail=1 --disable-warnings'
                        }
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Cleaning up workspace and Docker images...'
            // Docker cleanup can be done using the Docker pipeline syntax or PowerShell/bat
            bat 'docker system prune -f'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs for errors.'
        }
    }
}
