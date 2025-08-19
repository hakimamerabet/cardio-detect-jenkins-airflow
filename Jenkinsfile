pipeline {
    agent any

    environment {
        IMAGE_NAME = 'ml-pipeline-image'
        WORKSPACE_DIR = '/c/Users/hakima/.jenkins/workspace/ML-tests'
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/JedhaBootcamp/sample-ml-workflow.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${IMAGE_NAME}")
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
                        bat """
                        docker run --rm ^
                          -v ${WORKSPACE_DIR}:/workspace ^
                          -w /workspace ^
                          -e MLFLOW_TRACKING_URI ^
                          -e AWS_ACCESS_KEY_ID ^
                          -e AWS_SECRET_ACCESS_KEY ^
                          -e BACKEND_STORE_URI ^
                          -e ARTIFACT_ROOT ^
                          ${IMAGE_NAME} ^
                          sh -c "pytest --maxfail=1 --disable-warnings"
                        """
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Cleaning up workspace and Docker images...'
            bat 'docker system prune -f'
        }
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check logs for errors.'
        }
    }
}
