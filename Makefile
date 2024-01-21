PROD_COMPOSE   	:= ./docker-compose.yml

app.up:
	docker-compose -f ${PROD_COMPOSE} build && docker-compose -f ${PROD_COMPOSE} up -d;

app.down:
	docker-compose -f ${PROD_COMPOSE} down;

create-db:
	docker cp db-script.sql db:/db-script.sql && docker exec -u postgres db psql postgres postgres -f /db-script.sql
