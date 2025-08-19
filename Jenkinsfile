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
                        // Convert Windows-style WORKSPACE path to Docker-compatible format
                        def dockerPath = env.WORKSPACE.replaceAll('\\\\', '/').replaceAll('C:', '/c')

                        docker.image('ml-pipeline-image').inside(
                            "-v ${dockerPath}:/workspace " +
                            "-w /workspace " +
                            "-e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI " +
                            "-e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID " +
                            "-e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY " +
                            "-e BACKEND_STORE_URI=$BACKEND_STORE_URI " +
                            "-e ARTIFACT_ROOT=$ARTIFACT_ROOT"
                        ) {
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
